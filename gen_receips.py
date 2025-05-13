# Заполняем таблицу texts сгенерированными данными:
import sqlite3
import requests
from tqdm import tqdm
import re

OLLAMA_URL = "https://ollama.rety87nm.ru/"

# Сгенерировать рекомендацию
def gen_recomend(prompt, model="llama3:latest"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.5,
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error during translation: {e}")
        return ""

conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Заглушка для рецепта бла короткая не более 80 байтов, считаем что age_part, gender и dom_type - всегда заполнены пользователем
cursor.execute("select id, age_part, gender, allergen, dom_type, text from texts where length(text) <= 80 and age_part is not null and gender is not null and dom_type is not null")
rows = cursor.fetchall()
age_dict = {
        1:"до 20",
        2:"от 20 до 40",
        3:"от 40 до 60",
        4:"от 60 до 80",
        5:"старше 80",
}

# функция проверки есть ли англоязычная генерация
def has_english(text):
    if not isinstance(text, str):
        return False

    pattern = r'\b[a-zA-Z]{2,}\b'

    matches = re.findall(pattern, text)

    return len(matches) > 3

for r in tqdm(rows):
    user_info = []
    (id, age_part, gender, allergen, dom_type, old_text) = r

    if (gender):
        user_info.append( "Мужчина" if gender == 'муж' else "Женщина")

    if (age_part):
        user_info.append("возраст " + age_dict[age_part] + " лет")

    if (allergen):
        user_info.append(f"аллергия на {allergen}")

    if (dom_type):
        user_info.append(f"тип кожи - {dom_type}")
    
    if len(user_info) == 0:
        user_info = ["Европеец"]
    
    user_info = ", ".join(user_info)

    promt = f"Напиши рекомендации за косметическим уходом за кожей лица для пользователя на основе информации о нем. Информация о пользователе: {user_info}. Учитывай возрастные, половые и аллергические особенности и тип кожи если такие указаны. Ответ должен быть только на русском языке, использую только кирилицу. Не упоминай запрос и личные данные пользователя. Отвечай как косметолог. Используй только простое форматирование. Напиши только рекомендации без заголовка. Повторно проверь не используешь ли ты латиницу. Ответ СТРОГО без истользования латинских слов"

    gen_text = gen_recomend(promt)

    if( has_english(gen_text) ):
        print("Генерим повторно...")
        gen_text = gen_recomend(promt)
        print( has_english(gen_text) )

    print(f"Длина gen_text: {len(gen_text)}")
    print(f"Содержимое gen_text: {repr(gen_text)}")

    cursor.execute('update texts set text = ? where id = ?', (gen_text, id) )

    conn.commit()

conn.close()
