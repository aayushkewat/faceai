import cv2
import os
import numpy as np

def create_dataset(data_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_data = []
    labels = []
    label_dict = {}

    for label, person_name in enumerate(os.listdir(data_path)):
        person_path = os.path.join(data_path, person_name)
        label_dict[label] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_data.append(face)
                labels.append(label)

    return face_data, labels, label_dict

data_path = 'training_data'
face_data, labels, label_dict = create_dataset(data_path)

# Train the LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_data, np.array(labels))

# Save the model
face_recognizer.save('face_recognizer.yml')
np.save('label_dict.npy', label_dict)
