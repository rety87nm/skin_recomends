from PIL import Image
import os
import sqlite3
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from auth import get_user_by_username, check_password, get_user
from werkzeug.utils import secure_filename
import time
import config

from SkinTypeChecker import SkinTypeChecker

app = Flask(__name__)
app.secret_key = os.urandom(24)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Пожалуйста, войдите, чтобы получить доступ к этой странице."

sc = SkinTypeChecker(config.model_path)

UPLOAD_FOLDER = 'temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return get_user(user_id)

def age_parts(age):
    age = int(age)
    if age < 20:
        return 1
    elif age < 40:
        return 2
    elif age < 60:
        return 3
    elif age < 80:
        return 4
    else:
        return 5

def load_history(cursor, user_id):
    # Получаем 10 последних сообщений
    cursor.execute('''
        SELECT ar.id, ar.age, ar.gender, ar.allergies, ar.label, ar.probs, ar.filename, t.text
        FROM analysis_results as ar
        JOIN texts as t ON t.id = ar.text_id
        WHERE ar.user_id = ?
        ORDER BY ar.id DESC
        LIMIT 10
    ''', (user_id,))

    rows = cursor.fetchall()
    history = [
        {
            'id': row[0],
            'age': row[1],
            'gender': row[2],
            'allergies': row[3],
            'label': row[4],
            'probs': row[5],
            'filename': row[6],
            'receipt': row[7]
        }
        for row in rows
    ]
    return history

@app.route('/')
@login_required
def index():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    h = load_history(cursor, current_user.id)
    conn.commit()
    return render_template('index.html', history=h)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            flash('Пользователь с таким логином или email уже существует.','error')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                       (username, email, hashed_password))
        conn.commit()
        conn.close()

        user = get_user_by_username(username)
        if user and check_password(user, password):
            login_user(user, remember=True)
            return redirect(url_for('index'))


    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_ = request.form['login']
        password = request.form['password']

        user = get_user_by_username(login_)
        if user and check_password(user, password):
            login_user(user, remember=True)
            return redirect(url_for('index'))
        else:
            flash('Неверный логин или пароль','error')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'Не загружен файл изображения'}), 400

    # Генерация уникального имени файла с использованием времени
    timestamp = int(time.time())
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Открытие изображения и конвертация в JPG
    try:
        img = Image.open(filepath)
        jpg_filename = f"{timestamp}_{os.path.splitext(file.filename)[0]}.jpg"
        jpg_filepath = os.path.join(UPLOAD_FOLDER, jpg_filename)
        img.convert('RGB').save(jpg_filepath, 'JPEG')

        thumbnail = img.copy()
        thumbnail.thumbnail((64, 64))
        thumbnail_dir = os.path.join('static', 'img')
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_filename = f"thumb_{jpg_filename}"
        thumbnail_filepath = os.path.join(thumbnail_dir, thumbnail_filename)
        thumbnail.save(thumbnail_filepath, 'JPEG')

    except Exception as e:
        return jsonify({'error': f'Ошибка при обработке изображения: {str(e)}'}), 500

    return jsonify({
        'filename': jpg_filename,
        'thumbnail': thumbnail_filename
    })

@app.route('/analyze', methods=['POST'])
@login_required
def handler():
    try:
        age = request.form.get('age')
        gender = request.form.get('gender')
        allergies = request.form.get('allergies')
        filename = request.form.get('filename')

        if not all([age, gender, allergies, filename]):
            return jsonify({'error': 'Не все данные заполнены'}), 400

        #Анализ фотографии на тип кожи
        
        label, class_id, probs = sc.analyze('temp/' + filename)
        ru_label = sc.ru_classes[class_id]

        probs_str = ', '.join([f" {sc.ru_classes[class_id]} {int(p*100)}%" for class_id, p in enumerate(probs)]) if hasattr(probs, '__iter__') else str(probs)

        #sqlite
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()

        Q = 'SELECT id FROM texts where age_part = ? and gender = ? and dom_type = ?';
        P = [age_parts(age), gender, ru_label]
        
        if allergies == 'нет':
            Q += ' and allergen is NULL'
        else:
            Q += ' and allergen = ? '
            P.append(allergies)

        cursor.execute(Q, P)
        rows = cursor.fetchall()
        text_id = int( rows[0][0] )

        cursor.execute('''
                    INSERT INTO analysis_results
                    (label, probs, age, gender, allergies, filename, text_id, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (ru_label, probs_str, age, gender, allergies, filename, text_id, current_user.id))

        conn.commit()

        # Получаем 10 последних сообщений 
        history = load_history(cursor, current_user.id)

        result = {
            'label': ru_label,
            'probs': probs_str,
            'receipt': history[0]['receipt'].replace("\n", "<br>"),
            'history': history
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=4100)
