import os
import cv2
import numpy as np
from PIL import Image

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'


def get_images_with_id(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for single_image_path in image_paths:
        faceImg = Image.open(single_image_path).convert('L')  # convert to grayscale
        faceNp = np.array(faceImg, np.uint8)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(f"Processing ID: {id}")
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)

    return np.array(ids), faces


# Train the model
ids, faces = get_images_with_id(path)
print("Unique IDs found:", np.unique(ids))
recognizer.train(faces, ids)
os.makedirs("recognizer", exist_ok=True)
recognizer.save("recognizer/trainingdata.yml")
cv2.destroyAllWindows()
print("✅ Training complete and model saved successfully!")
