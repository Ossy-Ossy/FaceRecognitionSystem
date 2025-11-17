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

# ---------- IMPROVED FACE CAPTURE ----------
def save_face_samples(Id, img_file, num_samples=100):
    os.makedirs("dataset", exist_ok=True)
    image = Image.open(img_file)
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) == 0:
        st.warning("⚠️ No face detected. Try again with better lighting.")
        return
    
    # Take the largest face detected
    (x, y, w, h) = max(faces, key=lambda rect: rect[2]*rect[3])
    
    # IMPROVED: Apply preprocessing similar to recognition
    face_roi = gray[y:y+h, x:x+w]
    
    # Check if face is too small
    if w < 80 or h < 80:
        st.warning("⚠️ Face too small. Move closer to the camera.")
        return
    
    saved_count = 0
    for i in range(num_samples):
        # Create variations for robust training
        augmented = face_roi.copy()
        
        # Random brightness adjustment (simulate lighting changes)
        brightness = random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * brightness, 0, 255).astype(np.uint8)
        
        # Random small rotations (-10 to +10 degrees)
        angle = random.uniform(-10, 10)
        center = (augmented.shape[1] // 2, augmented.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (augmented.shape[1], augmented.shape[0]))
        
        # Random crops (simulate distance changes)
        crop_percent = random.uniform(0.85, 1.0)
        new_w = int(augmented.shape[1] * crop_percent)
        new_h = int(augmented.shape[0] * crop_percent)
        start_x = random.randint(0, max(1, augmented.shape[1] - new_w))
        start_y = random.randint(0, max(1, augmented.shape[0] - new_h))
        augmented = augmented[start_y:start_y+new_h, start_x:start_x+new_w]
        
        # Histogram equalization for better contrast
        augmented = cv2.equalizeHist(augmented)
        
        # Resize to standard size
        resized = cv2.resize(augmented, (200, 200))
        
        filename = f"dataset/user.{Id}.{i+1}.jpg"
        cv2.imwrite(filename, resized)
        saved_count += 1
    
    st.success(f"✅ Saved {saved_count} enhanced face samples for ID {Id}!")
    st.info("💡 Tip: Capture multiple photos in different lighting for best results.")

# ---------- STREAMLIT APP ----------
st.title("📸 Enhanced Face Registration System")
st.write("Capture your face with improved preprocessing for better recognition.")

create_table()

Id = st.number_input("Enter User ID", min_value=1, step=1)
Name = st.text_input("Enter Name")
Age = st.number_input("Enter Age", min_value=1, step=1)
MatricNo = st.text_input("Enter Matriculation Number (e.g. 2021/246553)")

if Name and MatricNo:
    insert_or_update(Id, Name, Age, MatricNo)
    
    st.info("📸 **Important Tips for Best Results:**")
    st.write("✓ Ensure good lighting on your face")
    st.write("✓ Face the camera directly")
    st.write("✓ Remove glasses if possible")
    st.write("✓ Keep a neutral expression")
    
    img_file = st.camera_input("📷 Capture your face")
    
    if img_file is not None:
        with st.spinner("Processing and saving 100 enhanced samples..."):
            save_face_samples(Id, img_file, num_samples=100)
else:
    st.warning("Please fill all fields before capturing your face.")
