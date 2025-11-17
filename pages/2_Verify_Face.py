import streamlit as st
import cv2
import sqlite3
import numpy as np
from PIL import Image

# -------------------- CONFIG --------------------
CASCADE_PATH = "haarcascade_frontalface_default.xml"
RECOGNIZER_PATH = "recognizer/trainingdata.yml"

# CRITICAL FIX: Lower threshold for LBPH (lower = stricter)
# Typical good match: 30-50, Acceptable: 50-70, Poor: 70+
CONFIDENCE_THRESHOLD = 50  # Adjust this: 40-60 is recommended

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
    
    if len(faces) == 0:
        st.warning("😕 No face detected. Try again.")
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Recognize the face
            id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            
            # CRITICAL FIX: For LBPH, LOWER confidence = BETTER match
            # Reject if confidence is too HIGH (meaning poor match)
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
                    cv2.putText(image, f"Conf: {conf:.1f}", (x, start_y + 75),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)
                    st.success(f"✅ Recognized: {profile[1]} (Age: {profile[2]}, RegNo: {profile[3]}) — Confidence: {conf:.2f}")
                else:
                    cv2.putText(image, "No Record Found", (x, y + h + 25),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 165, 255), 2)
                    st.warning(f"⚠️ ID={id} recognized but no record found in database.")
            else:
                # FIXED: Properly reject unknown faces
                cv2.putText(image, "UNKNOWN", (x, y + h + 25),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"Conf: {conf:.1f}", (x, y + h + 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                st.error(f"🚫 Unknown face detected. Confidence too low: {conf:.2f} (threshold: {CONFIDENCE_THRESHOLD})")
    
    # Display final image with overlays
    st.image(Image.fromarray(image), caption="Recognition Result", use_column_width=True)
    
    # Add diagnostic info
    with st.expander("🔍 Diagnostic Information"):
        st.write(f"**Faces detected:** {len(faces)}")
        st.write(f"**Confidence threshold:** {CONFIDENCE_THRESHOLD}")
        st.write("**Note:** For LBPH, lower confidence = better match")
        st.write("**Recommended threshold range:** 40-60")
        st.write("**Adjust threshold if:**")
        st.write("- Too many false rejections → increase threshold (e.g., 60-70)")
        st.write("- Too many false acceptances → decrease threshold (e.g., 40-45)")
