--DROP TABLE analysis_results;
--DROP TABLE texts;
--DROP INDEX idx_texts_search;
--DROP TABLE dict_alergens;
--DROP TABLE dict_gender;
--DROP TABLE dict_skin_types;
--DROP TABLE dict_age_parts;

BEGIN TRANSACTION;
CREATE TABLE dict_alergens(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO dict_alergens (name) VALUES(NULL);
INSERT INTO dict_alergens (name) VALUES('эфирные масла');
INSERT INTO dict_alergens (name) VALUES('растительные экстракты');
INSERT INTO dict_alergens (name) VALUES('отдушки и парфюмерные композиции');
INSERT INTO dict_alergens (name) VALUES('парабены');
INSERT INTO dict_alergens (name) VALUES('красители');
INSERT INTO dict_alergens (name) VALUES('ланолин');
INSERT INTO dict_alergens (name) VALUES('силиконы');
INSERT INTO dict_alergens (name) VALUES('минеральные масла');
INSERT INTO dict_alergens (name) VALUES('кислоты');
INSERT INTO dict_alergens (name) VALUES('продукты пчеловодства');

CREATE TABLE dict_gender(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO dict_gender (name) VALUES(NULL);
INSERT INTO dict_gender (name) VALUES('муж');
INSERT INTO dict_gender (name) VALUES('жен');

CREATE TABLE dict_skin_types(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
INSERT INTO dict_skin_types (name) VALUES(NULL);
INSERT INTO dict_skin_types (name) VALUES('Акне');
INSERT INTO dict_skin_types (name) VALUES('Сухая');
INSERT INTO dict_skin_types (name) VALUES('Жирная');
INSERT INTO dict_skin_types (name) VALUES('Нормальная');
INSERT INTO dict_skin_types (name) VALUES('Комбинированная');

CREATE TABLE dict_age_parts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name INTEGER
);
INSERT INTO dict_age_parts (name) VALUES(NULL);
INSERT INTO dict_age_parts (name) VALUES("1");
INSERT INTO dict_age_parts (name) VALUES("2");
INSERT INTO dict_age_parts (name) VALUES("3");
INSERT INTO dict_age_parts (name) VALUES("4");
INSERT INTO dict_age_parts (name) VALUES("5");

-- Таблица с данными пользователей и их результаты по уходу
CREATE TABLE analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER,
                gender TEXT,
                allergies TEXT,
                label TEXT,
                probs TEXT,
                filename TEXT,
                text_id INTEGER NOT NULL,
                FOREIGN KEY (text_id) REFERENCES texts(id)
            );

-- Таблица с предзагруженными рекомендациями по всем возможным вариантам
CREATE TABLE texts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- возрастной интервал 1 - 0..20, 2 - 20..40 ...  5 - 80..~
    age_part INTEGER,
    gender TEXT,
    allergen TEXT,
    -- доминирующий тип кожи
    dom_type TEXT,
    -- Текст рекомендации по уходу за кожей
    text TEXT
);

CREATE INDEX idx_texts_search ON texts (age_part, gender, allergen, dom_type);

# Декартово произведение всех значений заносим в таблицу тектов, далее скриптом обновим эти значения на более информативные:
INSERT INTO texts (age_part, gender, allergen, dom_type, text)
SELECT ap.name AS age_part,
       g.name AS gender,
       a.name AS allergen,
       st.name AS dom_type,
       'Рецепт по уходу за кожей для ' || COALESCE(g.name, '') || ' с алергией на: ' || COALESCE(a.name, '') AS text
FROM dict_age_parts AS ap CROSS JOIN dict_alergens AS a CROSS JOIN dict_gender AS g CROSS JOIN dict_skin_types AS st;

COMMIT;
