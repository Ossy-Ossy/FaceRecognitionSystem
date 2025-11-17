import streamlit as st
import cv2
import os
import sqlite3

# ---------- DATABASE ----------
def create_table():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS STUDENTS (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            Age INTEGER NOT NULL,
            MatricNo TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_data(name, age, matric):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO STUDENTS (Name, Age, MatricNo) VALUES (?, ?, ?)",
                   (name, age, matric))
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id

# ---------- FACE CAPTURE ----------
def capture_faces(user_id):
    st.write("📸 Capture 50 face samples. Click the camera button repeatedly.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    count = 0

    while count < 50:
        img = st.camera_input(f"Image {count+1}/50")

        if img is None:
            st.warning("Waiting for camera input...")
            st.stop()

        # Convert image
        image = np.array(Image.open(img))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.error("❌ No face detected. Try again.")
            continue

        # Only use largest face
        (x, y, w, h) = max(faces, key=lambda a: a[2] * a[3])
        face_img = gray[y:y+h, x:x+w]

        # Save sample
        cv2.imwrite(f"dataset/User.{user_id}.{count+1}.jpg", face_img)
        count += 1

        st.success(f"Saved sample {count}/50")

    st.success("✔ All 50 face samples captured!")


# ---------- STREAMLIT ----------
st.title("📝 Register User & Capture Face")

create_table()

name = st.text_input("Full Name")
age = st.number_input("Age", min_value=1, max_value=100)
matric = st.text_input("Matric Number (e.g., 2021/123456)")

if st.button("Register & Capture Face"):
    if name and age and matric:
        user_id = insert_data(name, age, matric)
        st.success(f"User Registered! Assigned ID = {user_id}")
        capture_faces(user_id)
    else:
        st.error("Fill all the fields first!")


