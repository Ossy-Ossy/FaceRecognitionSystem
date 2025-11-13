import streamlit as st
import cv2
import sqlite3
import numpy as np
from PIL import Image

# -------------------- CONFIG --------------------
CASCADE_PATH = "haarcascade_frontalface_default.xml"
RECOGNIZER_PATH = "recognizer/trainingdata.yml"
CONFIDENCE_THRESHOLD = 80  # Lower is stricter

# -------------------- INITIALIZE --------------------
facedetect = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(RECOGNIZER_PATH)

st.title("🎯 Real-time Face Recognition")
st.write("Click below to capture your face and verify your identity.")

# -------------------- DATABASE FUNCTION --------------------
def get_profile(student_id):
    """Fetch student info from database by ID."""
    conn = sqlite3.connect('database.db')
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE Id = ?", (student_id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

# -------------------- CAMERA INPUT --------------------
uploaded_image = st.camera_input("📸 Capture your face for recognition")

if uploaded_image is not None:
    # Convert uploaded image to OpenCV format
    image = np.array(Image.open(uploaded_image))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    recognized = False

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize the face
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])

        if conf < CONFIDENCE_THRESHOLD:
            profile = get_profile(id)
            if profile:
                recognized = True
                start_y = y + h + 25
                cv2.putText(image, f"Name: {profile[1]}", (x, start_y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 127), 2)
                cv2.putText(image, f"Age: {profile[2]}", (x, start_y + 25),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 127), 2)
                cv2.putText(image, f"RegNo: {profile[3]}", (x, start_y + 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 127), 2)

                st.success(f"✅ Recognized: {profile[1]} (Age: {profile[2]}, RegNo: {profile[3]}) — Confidence: {conf:.2f}")

            else:
                cv2.putText(image, "No Record Found", (x, y + h + 25),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 165, 255), 2)
                st.warning(f"⚠️ ID={id} recognized but no record found in database.")
        else:
            cv2.putText(image, "Unknown", (x, y + h + 25),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            st.error(f"🚫 Unknown face detected. Confidence={conf:.2f}")

    if not recognized and len(faces) == 0:
        st.warning("😕 No face detected. Try again.")

    # Display final image with overlays
    st.image(Image.fromarray(image), caption="Recognition Result", use_column_width=True)
