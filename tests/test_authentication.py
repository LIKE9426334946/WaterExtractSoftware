"""账号持久化、登录会话及工作区接口鉴权。"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import create_app
from utils.authentication import AccountStore, SESSION_COOKIE
from utils.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]


class AccountTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "users.json"
        self.store = AccountStore(self.path, 60)

    def test_default_and_registered_accounts_survive_restart_with_hashed_passwords(self):
        self.assertTrue(self.store.authenticate("admin", "123456"))
        self.assertFalse(self.store.authenticate("admin", "wrongpass"))
        self.store.register("student", "123456", "123456")
        saved = self.path.read_text()
        self.assertNotIn("123456", saved)
        records = json.loads(saved)
        self.assertNotEqual(records["admin"]["salt"], records["student"]["salt"])
        self.assertNotEqual(records["admin"]["password_hash"], records["student"]["password_hash"])
        reloaded = AccountStore(self.path, 60)
        self.assertTrue(reloaded.authenticate("student", "123456"))
        self.assertTrue(reloaded.authenticate("admin", "123456"))

    def test_invalid_registration_does_not_overwrite_accounts(self):
        before = self.path.read_bytes()
        for values in (("admin", "abcdef", "abcdef"), ("abc", "abcdef", "abcdefg"),
                       ("ab", "abcdef", "abcdef"), ("abc", "123", "123")):
            with self.assertRaises(ValueError):
                self.store.register(*values)
        self.assertEqual(self.path.read_bytes(), before)

    def test_session_expiration_logout_and_restart(self):
        with patch("utils.authentication.monotonic", return_value=100):
            token = self.store.start_session("admin")
            self.assertEqual(self.store.current_user(token), "admin")
        with patch("utils.authentication.monotonic", return_value=161):
            self.assertIsNone(self.store.current_user(token))
        token = self.store.start_session("admin")
        self.assertIsNone(AccountStore(self.path, 60).current_user(token))
        self.store.logout(token)
        self.assertIsNone(self.store.current_user(token))
        self.assertIsNone(self.store.current_user("forged-token"))


class AuthenticationHTTPTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = load_config(ROOT / "config.yaml")
        self.config = replace(base, users_file=Path(directory.name) / "users.json")
        self.client = self.enterContext(TestClient(create_app(self.config), follow_redirects=False))

    def test_login_protects_page_upload_queue_config_and_download_then_logout_revokes(self):
        self.assertEqual(self.client.get("/").headers["location"], "/login")
        self.assertEqual(self.client.get("/app/").headers["location"], "/login")
        for method, path in (("GET", "/app/config"), ("POST", "/app/gradio_api/upload"),
                             ("POST", "/app/gradio_api/queue/join"),
                             ("GET", "/app/gradio_api/file=anything.png")):
            self.assertEqual(self.client.request(method, path).status_code, 401)
        failed = self.client.post("/login", data={"username": "admin", "password": "wrongpass"})
        self.assertIn("账号或密码错误", failed.text)
        self.assertNotIn(SESSION_COOKIE, self.client.cookies)
        logged_in = self.client.post("/login", data={"username": "admin", "password": "123456"})
        self.assertEqual(logged_in.status_code, 303)
        self.assertEqual(logged_in.headers["location"], "/app/")
        self.assertIn("HttpOnly", logged_in.headers["set-cookie"])
        self.assertIn("SameSite=lax", logged_in.headers["set-cookie"])
        token = self.client.cookies[SESSION_COOKIE]
        self.assertEqual(self.client.get("/app/config").status_code, 200)
        self.assertIn(self.client.get("/app/gradio_api/file=" + str(self.config.users_file)).status_code, (403, 404))
        self.assertEqual(self.client.post("/logout").headers["location"], "/login")
        self.assertNotIn(SESSION_COOKIE, self.client.cookies)
        self.client.cookies.set(SESSION_COOKIE, token)
        self.assertEqual(self.client.get("/app/config").status_code, 401)

    def test_registration_errors_success_login_and_background(self):
        page = self.client.get("/login")
        self.assertIn('href="/register"', page.text)
        self.assertEqual(page.headers["cache-control"], "no-store")
        background = self.client.get("/auth/login.png")
        self.assertEqual(background.content, (ROOT / "ui/login.png").read_bytes())
        self.assertIn("/auth/login.png", self.client.get("/auth/login.css").text)
        fields = {"username": "student", "password": "pass123", "confirmation": "wrong123"}
        self.assertIn("两次输入的密码不一致", self.client.post("/register", data=fields).text)
        fields["confirmation"] = fields["password"]
        self.assertEqual(self.client.post("/register", data=fields).status_code, 303)
        self.assertIn("该账号已存在", self.client.post("/register", data=fields).text)
        login = self.client.post("/login", data={"username": "student", "password": "pass123"})
        self.assertEqual(login.status_code, 303)
        self.assertEqual(self.client.get("/app/config").status_code, 200)

    def test_cross_origin_forms_and_html_in_username(self):
        for path in ("/login", "/register", "/logout"):
            response = self.client.post(path, headers={"Origin": "https://elsewhere.example"})
            self.assertEqual(response.status_code, 403)
        response = self.client.post("/login", data={"username": '<script>alert(1)</script>', "password": "wrongpass"})
        self.assertNotIn('<script>alert(1)</script>', response.text)
        self.assertIn("&lt;script&gt;", response.text)


if __name__ == "__main__":
    unittest.main()
