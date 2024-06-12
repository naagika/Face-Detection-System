import cv2
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os




def clear_folder(folder, exclude_files=None, exclude_folders=None):
    """ Удаляет все файлы в указанной папке, исключая те, что указаны в списке exclude_files и exclude_folders. """
    if exclude_files is None:
        exclude_files = []
    if exclude_folders is None:
        exclude_folders = []

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isdir(file_path):
            if filename not in exclude_folders:
                clear_folder(file_path)  # рекурсивное удаление содержимого подпапок
                os.rmdir(file_path)  # удаление самой подпапки
        elif os.path.isfile(file_path) or os.path.islink(file_path):
            if filename not in exclude_files:
                os.unlink(file_path)


def get_latest_file(directory):
    """ Возвращает путь к последнему измененному файлу в указанной директории """
    # Получение списка файлов в директории
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    # Фильтрация списка только по файлам
    files = [f for f in files if os.path.isfile(f)]
    # Сортировка списка файлов по времени последней модификации
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def rotate_image(image, angle):
    """ Поворачивает изображение на указанный угол """
    (h, w) = image.shape[:2]  # получение размеров изображения
    center = (w // 2, h // 2)  # нахождение центра изображения
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # матрица поворота
    rotated = cv2.warpAffine(image, M, (w, h))  # применение поворота
    return rotated


# Путь к папке с фотографиями
photo_folder = 'camera'
# Папка, где будут сохраняться обработанные изображения
output_folder = 'camera/binary'

# Очищаем папку перед началом обработки
clear_folder(output_folder)


# Загрузка алгоритма Haar Cascade
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)


# Нахождение самой последней фотографии
latest_photo = get_latest_file(photo_folder)
latest_photo_name = os.path.basename(latest_photo)
# Чтение и поворот изображения
img = cv2.imread(latest_photo)
img = rotate_image(img, 90)  # Поворот на 90 градусов против часовой стрелки
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100)
)

# Обработка обнаруженных лиц
i = 0
for x, y, w, h in faces:
    cropped_image = gray_img[y:y + h, x:x + w]
    target_file_name = os.path.join(output_folder, f'{i}.jpg')
    cv2.imwrite(target_file_name, cropped_image)
    i += 1

print(f"Processed {latest_photo}, found {i} faces.")

print("binary partition finished")
clear_folder(photo_folder, exclude_files=[latest_photo_name], exclude_folders=['binary'])

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
