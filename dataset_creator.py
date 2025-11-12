import cv2
import sqlite3
import os

# ---------- DATABASE FUNCTIONS ----------

def create_table():
    """Ensure the STUDENTS table exists."""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS STUDENTS (
            Id INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Age INTEGER NOT NULL,
            MatricNo TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def insert_or_update(Id, Name, Age, MatricNo):
    """Insert or update a student's record."""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM STUDENTS WHERE Id = ?", (Id,))
    data = cursor.fetchone()

    if data:
        cursor.execute("""
            UPDATE STUDENTS SET Name=?, Age=?, MatricNo=? WHERE Id=?
        """, (Name, Age, MatricNo, Id))
        print(f"🔁 Updated record for ID {Id}")
    else:
        cursor.execute("""
            INSERT INTO STUDENTS (Id, Name, Age, MatricNo)
            VALUES (?, ?, ?, ?)
        """, (Id, Name, Age, MatricNo))
        print(f"✅ Inserted new record for ID {Id}")

    conn.commit()
    conn.close()


# ---------- FACE CAPTURE FUNCTION ----------

def capture_faces(Id, name, age, matric):
    """Open camera, detect face, draw rectangle, and save samples."""
    # Initialize camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Cannot open camera")
        return

    # Load face detector
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    os.makedirs("dataset", exist_ok=True)

    sampleNum = 0
    print("\n📸 Starting face capture... Look at the camera. Press 'q' to quit.\n")

    # Create and name the window so it shows properly
    cv2.namedWindow("Face Capture", cv2.WINDOW_NORMAL)

    while True:
        ret, img = cam.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            # Save cropped grayscale face
            cv2.imwrite(f"dataset/user.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])

            # Draw green rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Pause slightly between captures
            cv2.waitKey(100)

        # Show the camera feed with rectangles
        cv2.imshow("Face Capture", img)

        # Keep window responsive
        if cv2.getWindowProperty("Face Capture", cv2.WND_PROP_VISIBLE) < 1:
            break

        # Stop if 'q' is pressed or enough samples collected
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting face capture...")
            break
        if sampleNum >= 90:
            print(f"✅ Collected 90 samples for ID {Id}")
            break

    # Release resources
    cam.release()
    cv2.destroyAllWindows()
    print("📦 Face samples saved successfully!")


# ---------- MAIN PROGRAM ----------

def main():
    create_table()

    Id = int(input('Enter User ID: '))
    Name = input('Enter User Name: ')
    Age = int(input('Enter User Age: '))
    MatricNo = input('Enter Matriculation Number (e.g., 2021/246553): ')

    insert_or_update(Id, Name, Age, MatricNo)
    capture_faces(Id, Name, Age, MatricNo)


if __name__ == "__main__":
    main()
