import os
from PIL import Image
from imgbeddings import imgbeddings
import psycopg2

# Путь к папке с обработанными изображениями
input_folder = "camera/binary"

# Подключение к базе данных
conn = psycopg2.connect(
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Перебор всех файлов в папке
for filename in os.listdir(input_folder):
    # Полный путь к файлу
    file_path = os.path.join(input_folder, filename)

    # Загрузка и обработка изображения
    img = Image.open(file_path)
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(img)
    string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"

    # Поиск в базе данных
    cur.execute("SELECT name FROM knownfaces ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
    row = cur.fetchone()
    if row:
        print(f"The closest image to {filename} is {row[0]}")
    else:
        print(f"No match found for {filename}")

# Закрытие соединения
cur.close()
conn.close()
