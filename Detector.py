import cv2
import sqlite3
import os

# -------------------- CONFIG --------------------
CASCADE_PATH = "haarcascade_frontalface_default.xml"
RECOGNIZER_PATH = "recognizer/trainingdata.yml"
CONFIDENCE_THRESHOLD = 80  # Lower is stricter

# -------------------- INITIALIZE --------------------
facedetect = cv2.CascadeClassifier(CASCADE_PATH)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Camera not accessible.")
    exit()

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(RECOGNIZER_PATH)

# -------------------- DATABASE FUNCTION --------------------
def get_profile(student_id):
    """
    Fetch student info from the database by ID.
    Returns a tuple: (Id, Name, Age, MatricNo) or None
    """
    conn = sqlite3.connect('database.db')
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE Id = ?", (student_id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

# -------------------- MAIN LOOP --------------------
print("\n🎥 Starting face recognition... Press 'q' to quit.\n")

while True:
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize face
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])

        if conf < CONFIDENCE_THRESHOLD:
            profile = get_profile(id)
            if profile:
                # Display info on camera window
                start_y = y + h + 25
                cv2.putText(img, f"Name: {profile[1]}", (x, start_y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 127), 2)
                cv2.putText(img, f"Age: {profile[2]}", (x, start_y + 25),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 127), 2)
                cv2.putText(img, f"RegNo: {profile[3]}", (x, start_y + 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 127), 2)

                # Print info to terminal every time
                print(f"✅ Recognized: ID={profile[0]}, Name={profile[1]}, Age={profile[2]}, RegNo={profile[3]}, Conf={conf:.2f}")

            else:
                cv2.putText(img, "No Record Found", (x, y + h + 25),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 165, 255), 2)
                print(f"⚠️ ID={id} recognized but no record found in database. Conf={conf:.2f}")

        else:
            cv2.putText(img, "Unknown", (x, y + h + 25),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            print(f"🚫 Unknown face detected. Conf={conf:.2f}")

    cv2.imshow("FACE RECOGNITION", img)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- CLEAN UP --------------------
cam.release()
cv2.destroyAllWindows()
print("\n🛑 Recognition stopped.")
