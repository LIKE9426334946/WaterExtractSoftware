"""本地账号和登录会话；密码只保存随机盐及 PBKDF2 哈希。"""

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from threading import RLock
from time import monotonic

SESSION_COOKIE = "water_session"
PASSWORD_ITERATIONS = 600_000


def password_record(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PASSWORD_ITERATIONS
    ).hex()
    return {"salt": salt, "password_hash": digest}


class AccountStore:
    def __init__(self, path: Path, session_seconds: int):
        self.path = path
        self.session_seconds = session_seconds
        self.lock = RLock()
        self.sessions = {}
        if path.exists():
            self.users = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.users = {"admin": password_record("123456")}
            self.write_users(self.users)

    def write_users(self, users):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def register(self, username, password, confirmation):
        username = username.strip()
        if not 3 <= len(username) <= 32 or not all(c.isalnum() or c == "_" for c in username):
            raise ValueError("账号需为 3～32 位字母、数字、汉字或下划线。")
        if not 6 <= len(password) <= 128:
            raise ValueError("密码长度需为 6～128 位。")
        if password != confirmation:
            raise ValueError("两次输入的密码不一致。")
        with self.lock:
            if username in self.users:
                raise ValueError("该账号已存在，请使用其他账号或直接登录。")
            users = {**self.users, username: password_record(password)}
            self.write_users(users)
            self.users = users

    def authenticate(self, username, password):
        with self.lock:
            record = self.users.get(username.strip())
        if record is None or len(password) > 128:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(record["salt"]), PASSWORD_ITERATIONS
        ).hex()
        return hmac.compare_digest(record["password_hash"], digest)

    def start_session(self, username):
        token = secrets.token_urlsafe(32)
        now = monotonic()
        with self.lock:
            self.sessions = {key: value for key, value in self.sessions.items() if value[1] > now}
            self.sessions[token] = (username, now + self.session_seconds)
        return token

    def current_user(self, token):
        with self.lock:
            session = self.sessions.get(token)
            if session is not None and session[1] > monotonic():
                return session[0]
            self.sessions.pop(token, None)
        return None

    def logout(self, token):
        with self.lock:
            self.sessions.pop(token, None)
