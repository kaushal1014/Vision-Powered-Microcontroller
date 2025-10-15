import cv2
import urllib.request
import numpy as np
import face_recognition
import os

# Load all known face encodings from the "known_faces" folder
known_encodings = []
known_names = []

for filename in os.listdir("known_faces"):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        name = os.path.splitext(filename)[0]
        image = face_recognition.load_image_file(f"known_faces/{filename}")
        encs = face_recognition.face_encodings(image)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(name)

print(known_names)

# ESP32-CAM image URLs
LOW_URL = 'http://192.168.20.218/176x144.jpg'
HIGH_URL = 'http://192.168.20.218/320x240.jpg'

# Eye detection (optional visual)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cv2.namedWindow("Live Transmission", cv2.WINDOW_AUTOSIZE)

frame_count = 0
RECOGNITION_INTERVAL = 15
recognized_faces = []

while True:
    try:
        # Load low-res image for display
        img_resp = urllib.request.urlopen(LOW_URL)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
    except:
        print("Error loading low-res frame")
        continue

    frame_count += 1

    if frame_count % RECOGNITION_INTERVAL == 0:
        try:
            hi_resp = urllib.request.urlopen(HIGH_URL)
            hi_np = np.array(bytearray(hi_resp.read()), dtype=np.uint8)
            hi_img = cv2.imdecode(hi_np, -1)

            rgb_hi = cv2.cvtColor(hi_img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_hi)
            face_encodings = face_recognition.face_encodings(rgb_hi, face_locations)

            recognized_faces = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                if any(matches):
                    best_index = np.argmin(face_distances)
                    if face_distances[best_index] < 0.5:
                        name = known_names[best_index]


                # Scale down coordinates to low-res display
                scale_x = img.shape[1] / hi_img.shape[1]
                scale_y = img.shape[0] / hi_img.shape[0]
                left = int(left * scale_x)
                right = int(right * scale_x)
                top = int(top * scale_y)
                bottom = int(bottom * scale_y)

                recognized_faces.append(((left, top, right, bottom), name))

        except Exception as e:
            print("High-res recognition error:", e)

    # Draw labels and boxes on low-res stream
    for (left, top, right, bottom), name in recognized_faces:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Optional eye detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 1)

    cv2.imshow("Live Transmission", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
#test