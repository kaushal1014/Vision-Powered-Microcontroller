import face_recognition
import pickle
import cv2
import os

dataset_dir = "dataset"   # folder with subfolders of people
encodings_file = "encodings.pickle"

known_encodings = []
known_names = []

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"[INFO] processing {person}...")

    for img_file in os.listdir(person_dir):
        path = os.path.join(person_dir, img_file)
        image = cv2.imread(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect + encode
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person)

# save encodings to file
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] encodings saved to {encodings_file}")
