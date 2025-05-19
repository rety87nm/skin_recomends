import sqlite3

try:
    conn = sqlite3.connect('data.db')
    c = conn.cursor()

    #Создаем новую таблицу users
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT,
        password TEXT
    );
    """)

    #Добавляем поля в существующую таблицу analysis_results
    c.executescript("""
    CREATE TABLE IF NOT EXISTS analysis_results_dg_tmp
    (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        age       INTEGER,
        gender    TEXT,
        allergies TEXT,
        label     TEXT,
        probs     TEXT,
        filename  TEXT,
        text_id   INTEGER NOT NULL
            REFERENCES texts,
        user_id   INTEGER
            CONSTRAINT analysis_results_users_id_fk
                REFERENCES users
    );

    INSERT INTO analysis_results_dg_tmp(id, age, gender, allergies, label, probs, filename, text_id)
    SELECT id, age, gender, allergies, label, probs, filename, text_id
    FROM analysis_results;

    DROP TABLE analysis_results;

    ALTER TABLE analysis_results_dg_tmp
        RENAME TO analysis_results;
    """)

    conn.commit()
    conn.close()
    print("Миграция выполнена успешно.")

except Exception as e:
    print("Ошибка при выполнении миграции:", e)

finally:
    if conn:
        conn.close()
