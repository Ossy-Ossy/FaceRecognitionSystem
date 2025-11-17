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
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    count = 0
    st.write("📸 Capturing 50 face images... Please look at the camera.")

    while True:
        ret, frame = cam.read()
        if not ret:
            st.write("Camera not found!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) == 27 or count == 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    st.success("✔ Face capture completed!")

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
