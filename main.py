import tkinter as tk
from tkinter import messagebox, Canvas, Scrollbar
from PIL import Image, ImageTk, ImageSequence
import cv2
import os
import psycopg2
from imgbeddings import imgbeddings


def clear_folder(folder, exclude_files=None, exclude_folders=None):
    if exclude_files is None:
        exclude_files = []
    if exclude_folders is None:
        exclude_folders = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isdir(file_path) and filename not in exclude_folders:
            os.rmdir(file_path)
        elif os.path.isfile(file_path) and filename not in exclude_files:
            os.unlink(file_path)


def load_image_from_frame(frame, max_size=(300, 300)):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    original_size = image.size
    ratio = min(max_size[0] / original_size[0], max_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def update_stream():
    global latest_frame
    ret, latest_frame = camera.read()
    if ret:
        photo_image = load_image_from_frame(latest_frame, max_size=(800, 600))
        lbl_image.config(image=photo_image)
        lbl_image.image = photo_image
    root.after(10, update_stream)


def detect_faces():
    if latest_frame is None:
        messagebox.showerror("Error", "No photo found to process.")
        return

    lbl_status.config(text="Detecting faces...")
    show_loading()  # Показать иконку загрузки

    root.after(100, process_faces)


def show_loading():
    global loading_window, loading_label, frames, current_frame
    loading_window = tk.Toplevel(root)
    loading_window.geometry("256x256+{}+{}".format(root.winfo_x() + root.winfo_width() // 2 - 128, root.winfo_y() + root.winfo_height() // 2 - 128))
    loading_window.overrideredirect(True)
    loading_window.attributes("-topmost", True)
    loading_window.wm_attributes("-alpha", 0.5)  # Сделать окно полупрозрачным

    frames = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(loading_image)]
    loading_label = tk.Label(loading_window, bg="white")  # Установите цвет фона, который вы хотите сделать прозрачным
    loading_label.pack(expand=True)
    current_frame = 0
    animate_loading()


def animate_loading():
    global current_frame
    loading_label.config(image=frames[current_frame])
    current_frame = (current_frame + 1) % len(frames)
    loading_window.after(100, animate_loading)


def hide_loading():
    if loading_window:
        loading_window.destroy()


def process_faces():
    gray_img = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        lbl_status.config(text="No faces detected.")
        hide_loading()  # Убрать иконку загрузки
        return

    clear_folder(output_folder)
    for i, (x, y, w, h) in enumerate(faces):
        cropped_img = gray_img[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(output_folder, f"{i}.jpg"), cropped_img)

    lbl_status.config(text=f"Detected {len(faces)} faces.")
    show_matches()


def show_matches():
    global photo_images
    photo_images = []
    canvas.delete("all")
    conn = psycopg2.connect(dbname="postgres", user="nagima", password="root", host="localhost", port="5432")
    cur = conn.cursor()
    y_position = 10
    for filename in os.listdir(output_folder):
        img_path = os.path.join(output_folder, filename)
        img = Image.open(img_path)
        ibed = imgbeddings()
        embedding = ibed.to_embeddings(img)
        str_repr = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
        cur.execute("SELECT name FROM knownfaces ORDER BY embedding <-> %s LIMIT 1;", (str_repr,))
        row = cur.fetchone()
        match_name = row[0] if row else 'No match found'
        photo_image = load_image_from_frame(cv2.imread(img_path), (100, 100))
        photo_images.append(photo_image)
        canvas.create_image(10, y_position, anchor='nw', image=photo_image)
        canvas.create_text(320, y_position + 50, anchor='nw', text=match_name, font=('Arial', 12, 'bold'))
        y_position += 110
    canvas.configure(scrollregion=canvas.bbox('all'))
    cur.close()
    conn.close()
    hide_loading()
    lbl_status.config(text="Results displayed.")


root = tk.Tk()
root.title("Face Recognition System")

camera = cv2.VideoCapture(0)
latest_frame = None

photo_folder = 'camera'
output_folder = 'camera/binary'
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
canvas = Canvas(root, width=400, height=600)
canvas.pack(side="left", fill="both", expand=True)
scroll_y = Scrollbar(root, orient="vertical", command=canvas.yview)
scroll_y.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scroll_y.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
lbl_image = tk.Label(root)
lbl_image.pack()

btn_detect = tk.Button(root, text="Detect Faces", command=detect_faces)
btn_detect.pack()


loading_image = Image.open("loading.gif")
loading_window = None
frames = []

lbl_status = tk.Label(root, text="Streaming from camera...")
lbl_status.pack()

update_stream()
root.mainloop()

camera.release()
cv2.destroyAllWindows()


