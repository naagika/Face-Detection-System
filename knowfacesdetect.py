# importing the cv2 library
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os
import cv2

# Путь к папке с известными лицами и куда сохранять обработанные изображения
input_folder = "known-faces"
output_folder = "known-faces/binary"

# Загрузка алгоритма Haar Cascade
alg_path = r"haarcascade_frontalface_default.xml"  # Обновите путь к файлу
haar_cascade = cv2.CascadeClassifier(alg_path)


def clear_folder(folder):
    """ Удаляет все файлы в указанной папке """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

clear_folder(output_folder)

# Создание выходной папки, если она не существует
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Перебор всех файлов в папке с известными лицами
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # Проверка формата файла
        file_path = os.path.join(input_folder, filename)
        img = cv2.imread(file_path)
        if img is None:
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(
            gray_img, scaleFactor=1.05, minNeighbors=6, minSize=(100, 100)
        )

        # Обработка каждого обнаруженного лица
        base_name = os.path.splitext(filename)[0]  # Получение имени файла без расширения
        for i, (x, y, w, h) in enumerate(faces):
            cropped_image = gray_img[y:y+h, x:x+w]
            output_file_name = os.path.join(output_folder, f"{base_name}.jpg")
            cv2.imwrite(output_file_name, cropped_image)

        print(f"Processed {filename}, found {len(faces)} faces.")





# connecting to the database - replace the SERVICE URI with the service URI
conn = psycopg2.connect(
    dbname="postgres",
    user="nagima",
    password="root",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Очистка таблицы перед добавлением новых данных
cur.execute("DELETE FROM knownfaces;")
conn.commit()


for filename in os.listdir("known-faces/binary/"):
    # opening the image
    img = Image.open("known-faces/binary/" + filename)
    # loading the `imgbeddings`
    ibed = imgbeddings()
    # calculating the embeddings
    embedding = ibed.to_embeddings(img)
    filename = os.path.splitext(filename)[0]
    cur.execute("INSERT INTO knownfaces values (%s,%s)", (filename, embedding[0].tolist()))
    print(filename)
conn.commit()

