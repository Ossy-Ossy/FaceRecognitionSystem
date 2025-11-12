import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.title("🔹 Face Recognizer Training")

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)
os.makedirs("recognizer", exist_ok=True)

def get_images_with_id(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    faces = []
    ids = []

    for single_image_path in image_paths:
        faceImg = Image.open(single_image_path).convert('L')  # convert to grayscale
        faceNp = np.array(faceImg, np.uint8)
        try:
            id = int(os.path.split(single_image_path)[-1].split(".")[1])
        except ValueError:
            st.warning(f"Skipping file {single_image_path}, unable to extract ID")
            continue
        st.write(f"Processing ID: {id} -> {single_image_path}")
        faces.append(faceNp)
        ids.append(id)

    return np.array(ids), faces

# Button to trigger training
if st.button("Verify Face"):
    if not os.listdir(dataset_path):
        st.error("❌ Dataset folder is empty! Capture faces first.")
    else:
        ids, faces = get_images_with_id(dataset_path)
        if len(ids) == 0:
            st.error("❌ No valid face images found in dataset.")
        else:
            st.write(f"Unique IDs found: {np.unique(ids)}")
            recognizer.train(faces, ids)
            recognizer.save("recognizer/trainingdata.yml")
            st.success("✅ Training complete! Model saved to recognizer/trainingdata.yml")
