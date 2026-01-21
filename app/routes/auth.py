from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import login_user, logout_user, login_required, UserMixin
from app.extensions import login_manager

auth_bp = Blueprint('auth', __name__)

# 简单的用户类
class User(UserMixin):
    def __init__(self, id, role='user'):
        self.id = id
        self.role = role

USERS_DB = {'admin': {'password': 'admin', 'role': 'admin'}, 'user': {'password': 'user', 'role': 'user'}}

@login_manager.user_loader
def load_user(user_id):
    if user_id in USERS_DB:
        return User(user_id, USERS_DB[user_id]['role'])
    return None

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS_DB and USERS_DB[username]['password'] == password:
            user = User(username, USERS_DB[username]['role'])
            login_user(user)
            return redirect(url_for('dashboard.index'))
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))