import cv2
import face_recognition
import numpy as np
import pickle
import requests
from tensorflow.keras.models import load_model

# ESP32-CAM image URL
ESP32_URL = 'http://192.168.20.218/320x240.jpg'  # change resolution if needed

# Load label encoder (optional, only if used)
with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# Load liveness detection model (Keras H5)
liveness_model = load_model("liveness.model.h5")

# Load known face encodings + names
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)  # should contain { "encodings": [...], "names": [...] }

known_encodings = data["encodings"]
known_names = data["names"]

cv2.namedWindow("Face + Liveness Detection", cv2.WINDOW_AUTOSIZE)

while True:
    try:
        resp = requests.get(ESP32_URL, timeout=5)
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)


        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Only perform face recognition
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"
            if any(matches):
                best_index = np.argmin(face_distances)
                if face_distances[best_index] < 0.5:  # threshold
                    name = known_names[best_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face + Liveness Detection", frame)

    except Exception as e:
        print("Error fetching frame:", e)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
