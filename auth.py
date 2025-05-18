# auth.py

import sqlite3
from flask_login import UserMixin
from werkzeug.security import check_password_hash


class User(UserMixin):
    def __init__(self, user_id, username, email):
        self.id = user_id
        self.username = username
        self.email = email


def get_user(user_id):
    conn = sqlite3.connect('data.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user_data = cursor.fetchone()
        if user_data:
            return User(
                user_data["id"],
                user_data["username"],
                user_data["email"]
            )
    finally:
        conn.close()

    return None


def get_user_by_username(login):
    conn = sqlite3.connect('data.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (login, login))
        user_data = cursor.fetchone()
        if user_data:
            return User(
                user_data["id"],
                user_data["username"],
                user_data["email"]
            )
    finally:
        conn.close()

    return None


def check_password(user, password):
    conn = sqlite3.connect('data.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM users WHERE id = ?", (user.id,))
        user_data = cursor.fetchone()
        if user_data and check_password_hash(user_data["password"], password):
            return True
        return False
    finally:
        conn.close()