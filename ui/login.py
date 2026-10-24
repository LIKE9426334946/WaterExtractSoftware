"""登录、注册页面及 HTTP 路由；提取工作区继续使用 Gradio。"""

from html import escape
from pathlib import Path
from string import Template

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from utils.authentication import SESSION_COOKIE, AccountStore

UI_DIR = Path(__file__).resolve().parent
PAGE = Template((UI_DIR / "login.html").read_text(encoding="utf-8"))


def render_page(config, *, register=False, message="", username="", success=False):
    confirmation = """
        <label for="confirmation">确认密码</label>
        <input id="confirmation" name="confirmation" type="password" placeholder="请再次输入密码"
               autocomplete="new-password" minlength="6" maxlength="128" required>
    """ if register else ""
    notice = (
        f'<p class="notice {"success" if success else "error"}" role="alert">{escape(message)}</p>'
        if message else ""
    )
    html = PAGE.substitute(
        app_name=escape(config.name), version=escape(config.version),
        title="注册账号" if register else "欢迎登录",
        action="/register" if register else "/login", username=escape(username, quote=True),
        password_autocomplete="new-password" if register else "current-password",
        confirmation=confirmation, notice=notice, submit="注册" if register else "登录",
        switch=(
            '已有账号？<a href="/login">返回登录</a>' if register
            else '还没有账号？<a href="/register">立即注册</a>'
        ),
    )
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


def create_auth_app(config, accounts: AccountStore):
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def current_user(request: Request):
        return accounts.current_user(request.cookies.get(SESSION_COOKIE))

    @web.middleware("http")
    async def require_login(request: Request, call_next):
        path = request.url.path
        # 表单仅接受同站提交；同源的无 Origin 客户端请求仍可用于 API 调用。
        if request.method == "POST" and path in {"/login", "/register", "/logout"}:
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "请在本站页面提交表单。"}, status_code=403)
        # 同时保护页面、上传、推理、队列及下载，不能通过直接调用接口跳过登录。
        if path == "/app" or path.startswith("/app/"):
            if current_user(request) is None:
                if path in {"/app", "/app/"} and request.method == "GET":
                    return RedirectResponse("/login", status_code=303)
                return JSONResponse({"detail": "请先登录。"}, status_code=401)
        response = await call_next(request)
        if path == "/app" or path.startswith("/app/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @web.get("/")
    def home(request: Request):
        return RedirectResponse("/app/" if current_user(request) else "/login", status_code=303)

    @web.get("/login")
    def login_page(request: Request, registered: bool = False):
        if current_user(request):
            return RedirectResponse("/app/", status_code=303)
        return render_page(config, message="注册成功，请登录。" if registered else "", success=registered)

    @web.post("/login")
    def login(request: Request, username: str = Form(""), password: str = Form("")):
        if not accounts.authenticate(username, password):
            return render_page(config, message="账号或密码错误。", username=username)
        accounts.logout(request.cookies.get(SESSION_COOKIE))
        token = accounts.start_session(username.strip())
        response = RedirectResponse("/app/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE, token, max_age=accounts.session_seconds,
            httponly=True, samesite="lax", secure=request.url.scheme == "https",
        )
        return response

    @web.get("/register")
    def register_page(request: Request):
        if current_user(request):
            return RedirectResponse("/app/", status_code=303)
        return render_page(config, register=True)

    @web.post("/register")
    def register(username: str = Form(""), password: str = Form(""), confirmation: str = Form("")):
        try:
            accounts.register(username, password, confirmation)
        except ValueError as exc:
            return render_page(config, register=True, message=str(exc), username=username)
        return RedirectResponse("/login?registered=true", status_code=303)

    @web.post("/logout")
    def logout(request: Request):
        accounts.logout(request.cookies.get(SESSION_COOKIE))
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE)
        return response

    @web.get("/auth/login.png")
    def background():
        return FileResponse(UI_DIR / "login.png", media_type="image/png")

    @web.get("/auth/login.css")
    def stylesheet():
        return FileResponse(UI_DIR / "login.css", media_type="text/css")

    return web, current_user
