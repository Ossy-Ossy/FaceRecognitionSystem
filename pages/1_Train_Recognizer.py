import streamlit as st
import cv2
import os
import numpy as np

def train_model():
    dataset_dir = "dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            path = os.path.join(dataset_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            id = int(filename.split(".")[1])
            faces.append(img)
            ids.append(id)

    recognizer.train(faces, np.array(ids))
    recognizer.save("trainer.yml")

    return len(np.unique(ids))

st.title("🧠 Train Face Recognizer")

if st.button("Train Model"):
    count = train_model()
    st.success(f"✔ Training complete! Trained on {count} users.")
