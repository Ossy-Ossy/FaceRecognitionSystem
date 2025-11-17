 
import streamlit as st
import cv2
import numpy as np
import os
import sqlite3
from PIL import Image
import random

# ---------- DATABASE ----------
def create_table():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS STUDENTS (
            Id INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Age INTEGER NOT NULL,
            MatricNo TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_or_update(Id, Name, Age, MatricNo):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM STUDENTS WHERE Id=?", (Id,))
    data = cursor.fetchone()

    if data:
        cursor.execute("UPDATE STUDENTS SET Name=?, Age=?, MatricNo=? WHERE Id=?",
                       (Name, Age, MatricNo, Id))
        st.info(f"🔁 Updated record for ID {Id}")
    else:
        cursor.execute("INSERT INTO STUDENTS (Id, Name, Age, MatricNo) VALUES (?, ?, ?, ?)",
                       (Id, Name, Age, MatricNo))
        st.success(f"✅ New record created for ID {Id}")

    conn.commit()
    conn.close()


# ---------- FACE CAPTURE ----------
def save_face_samples(Id, img_file, num_samples=90):
    os.makedirs("dataset", exist_ok=True)
    image = Image.open(img_file)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    if len(faces) == 0:
        st.warning("⚠️ No face detected. Try again.")
        return

    # Take the **largest face** detected (usually the main user)
    (x, y, w, h) = max(faces, key=lambda rect: rect[2]*rect[3])

    for i in range(num_samples):
        # Apply small random shifts
        dx, dy = random.randint(-4, 4), random.randint(-4, 4)
        x1, y1 = max(0, x+dx), max(0, y+dy)
        x2, y2 = min(gray.shape[1], x+w+dx), min(gray.shape[0], y+h+dy)
        cropped = gray[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (200, 200))

        filename = f"dataset/user.{Id}.{i+1}.jpg"
        cv2.imwrite(filename, resized)

    st.success(f"✅ Saved {num_samples} face samples for ID {Id}!")

# ---------- STREAMLIT APP ----------
st.title("📸 Face Registration System")
st.write("Capture and register your face directly from your browser (90 samples per user).")

create_table()

Id = st.number_input("Enter User ID", min_value=1, step=1)
Name = st.text_input("Enter Name")
Age = st.number_input("Enter Age", min_value=1, step=1)
MatricNo = st.text_input("Enter Matriculation Number (e.g. 2021/246553)")

if Name and MatricNo:
    insert_or_update(Id, Name, Age, MatricNo)
    st.info("Click below to take a photo and generate 90 face samples.")

    img_file = st.camera_input("📷 Capture your face")

    if img_file is not None:
        with st.spinner("Processing and saving 90 samples..."):
            save_face_samples(Id, img_file, num_samples=90)
else:
    st.warning("Please fill all fields before capturing your face.")

