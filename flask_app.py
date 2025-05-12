from PIL import Image
import os
import sqlite3
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import time
import config

from SkinTypeChecker import SkinTypeChecker

app = Flask(__name__)

sc = SkinTypeChecker(config.model_path)

UPLOAD_FOLDER = 'temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'Не загружен файл изображения'}), 400

    #Генерация уникального имени файла с использованием времени
    timestamp = int(time.time())
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    #Открытие изображения и конвертация в JPG
    try:
        img = Image.open(file)
        jpg_filepath = os.path.splitext(filepath)[0] + '.jpg'
        img.convert('RGB').save(jpg_filepath, 'JPEG')
    except Exception as e:
        return jsonify({'error': f'Ошибка при обработке изображения: {str(e)}'}), 500

    return jsonify({'filename': jpg_filepath})

@app.route('/analyze', methods=['POST'])
def handler():
    try:
        age = request.form.get('age')
        gender = request.form.get('gender')
        allergies = request.form.get('allergies')
        filename = request.form.get('filename')

        if not all([age, gender, allergies, filename]):
            return jsonify({'error': 'Не все данные заполнены'}), 400

        #analyze
        label, class_id, probs = sc.analyze(filename)
        ru_label = sc.ru_classes[class_id]

        probs_str = ', '.join([f" {sc.ru_classes[class_id]} {int(p*100)}%" for class_id, p in enumerate(probs)]) if hasattr(probs, '__iter__') else str(probs)

        #sqlite
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER,
                gender TEXT,
                allergies TEXT,
                label TEXT,
                probs TEXT,
                filename TEXT
            )
        ''')
        cursor.execute(
            'INSERT INTO analysis_results (label, probs, age, gender, allergies, filename) VALUES (?, ?, ?, ?, ?, ?)',
            (ru_label, probs_str, age, gender, allergies, filename)
        )
        conn.commit()

        # Получаем всю историю
        cursor.execute('SELECT id, age, gender, allergies, label, probs FROM analysis_results ORDER BY id DESC')
        rows = cursor.fetchall()
        conn.close()

        history = [
            {
                'id': row[0],
                'age': row[1],
                'gender': row[2],
                'allergies': row[3],
                'label': row[4],
                'probs': row[5]
            }
            for row in rows
        ]

        result = {
            'label': ru_label,
            'probs': probs_str,
            'history': history
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=4100)
