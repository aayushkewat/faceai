import cv2
import numpy as np

# Load the pre-trained face detection model and LBPH face recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')
label_dict = np.load('label_dict.npy', allow_pickle=True).item()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face)

        if confidence < 100:
            person_name = label_dict[label]
            text = f"{person_name} ({confidence:.2f})"
        else:
            text = "Unknown"

        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Live Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
