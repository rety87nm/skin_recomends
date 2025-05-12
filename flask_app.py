import json
import os
import sqlite3
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from inference import analyze

app = Flask(__name__)
UPLOAD_FOLDER = 'temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    return jsonify({'filename': filepath})

@app.route('/analyze', methods=['POST'])
def handler():
    age = request.form.get('age')
    gender = request.form.get('gender')
    allergies = request.form.get('allergies')
    filename = request.form.get('filename')

    if not all([age, gender, allergies, filename]):
        return jsonify({'error': 'Missing form data'}), 400

    #analyze
    label, probs = analyze(filename)
    probs_str = ', '.join([f"{p:.2f}" for p in probs]) if hasattr(probs, '__iter__') else str(probs)


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
        (label, probs_str, age, gender, allergies, filename)
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
        'label': label,
        'probs': probs_str,
        'history': history
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=4100)
