# Vision-Powered-Microcontroller

Face Recognition System with ESP32-CAM Integration

## Overview

This project implements a real-time face recognition system that can work with both webcam and ESP32-CAM streams. It uses the `face_recognition` library for facial detection and recognition, with support for liveness detection using a trained deep learning model.

## Features

- Face detection and recognition using pre-encoded face embeddings
- ESP32-CAM integration for remote video streaming
- Liveness detection model (optional, can be disabled)
- Support for multiple known faces
- Real-time video processing

## Requirements

### Hardware
- Webcam (for local testing) or ESP32-CAM module
- Computer with Python 3.7+

### Software Dependencies

Install the required Python packages:

```bash
pip install face_recognition opencv-python numpy pickle-mixin requests tensorflow scikit-learn
```

**Note:** On Windows, you may need to install `cmake` and `dlib` separately:

```bash
pip install cmake
pip install dlib
```

## Project Structure

```
esp32_project/
├── encode_faces.py          # Script to generate face encodings from dataset
├── Stream.py               # Main script for ESP32-CAM face recognition
├── encodings.pickle        # Saved face encodings (generated)
├── label_encoder.pickle    # Label encoder for liveness model
├── liveness.model.h5       # Trained liveness detection model
├── dataset/               # Training images organized by person
│   ├── person1/
│   │   └── 1.jpg
│   ├── person2/
│   │   └── 1.jpeg
│   └── ...
└── README.md
```

## Setup Instructions

### Step 1: Prepare Your Dataset

1. Create a `dataset` folder in the project root
2. Inside `dataset`, create a subfolder for each person you want to recognize
3. Add face images (JPG/JPEG/PNG) to each person's folder

Example structure:
```
dataset/
├── kaushal/
│   └── 1.jpg
├── prashant/
│   └── 1.jpeg
└── veena/
    └── 1.jpeg
```

**Tips:**
- Use clear, well-lit face images
- Multiple images per person improve accuracy
- Images should primarily contain the person's face

### Step 2: Generate Face Encodings

Run the encoding script to process your dataset:

```bash
python encode_faces.py
```

This will:
- Process all images in the `dataset` folder
- Generate 128-dimensional face encodings for each detected face
- Save the encodings to `encodings.pickle`

### Step 3: Configure ESP32-CAM (Optional)

If using ESP32-CAM:

1. Flash your ESP32-CAM with camera streaming firmware
2. Connect the ESP32-CAM to your WiFi network
3. Note the IP address assigned to your ESP32-CAM
4. Update the `ESP32_URL` in `Stream.py`:

```python
ESP32_URL = 'http://YOUR_ESP32_IP/320x240.jpg'
```

**Common ESP32-CAM resolutions:**
- `160x120.jpg` (QQVGA)
- `320x240.jpg` (QVGA)
- `640x480.jpg` (VGA)
- `800x600.jpg` (SVGA)

### Step 4: Run Face Recognition

For ESP32-CAM streaming:

```bash
python Stream.py
```

The application will:
- Connect to the ESP32-CAM stream
- Detect faces in each frame
- Compare detected faces against known encodings
- Display the person's name or "Unknown" on the video feed

**Controls:**
- Press `q` to quit the application

## Face Recognition Settings

### Adjusting Recognition Threshold

In `Stream.py`, you can adjust the recognition sensitivity:

```python
if face_distances[best_index] < 0.5:  # Lower = stricter matching
    name = known_names[best_index]
```

- **Lower threshold (e.g., 0.4):** More strict, fewer false positives
- **Higher threshold (e.g., 0.6):** More lenient, may increase false positives

### Detection Model

The `face_recognition` library supports two detection models:

1. **HOG (Histogram of Oriented Gradients)** - Faster, CPU-friendly (default)
2. **CNN (Convolutional Neural Network)** - More accurate, GPU recommended

To change the model in `encode_faces.py`:

```python
boxes = face_recognition.face_locations(rgb, model="cnn")  # or "hog"
```

## Liveness Detection (Optional)

The liveness detection feature is currently disabled in `Stream.py`. It was designed to distinguish between real faces and spoofed images/videos.

### To Enable Liveness Detection:

Uncomment the liveness code in `Stream.py` and ensure you have:
- A trained `liveness.model.h5` file
- The corresponding `label_encoder.pickle`

The model expects input images of size (32, 32, 3).

### Training Your Own Liveness Model:

See the `face-recognition-with-liveness-web-login` directory for training scripts and datasets.

## Troubleshooting

### No faces detected
- Ensure images are clear and well-lit
- Try using `model="cnn"` for better detection
- Check that the camera is working properly

### "Unknown" displayed for known faces
- Add more training images per person
- Lower the recognition threshold (e.g., from 0.5 to 0.6)
- Ensure training images are similar to test conditions (lighting, angle)

### ESP32-CAM connection issues
- Verify the ESP32-CAM IP address is correct
- Check that both devices are on the same network
- Ensure the ESP32-CAM firmware is running properly
- Try pinging the ESP32-CAM IP address

### High CPU usage
- Use a lower resolution stream from ESP32-CAM
- Reduce the frame processing rate
- Use HOG model instead of CNN for face detection

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- On Windows, install Visual C++ build tools if `dlib` fails to install
- Consider using a virtual environment to avoid conflicts

## Performance Optimization

1. **Lower Resolution:** Use smaller image sizes (320x240 instead of 640x480)
2. **Skip Frames:** Process every 2nd or 3rd frame instead of every frame
3. **Detection Model:** Use HOG instead of CNN if you don't have a GPU
4. **Limit Known Faces:** Fewer known faces = faster comparison

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey
- ESP32-CAM community for camera streaming examples