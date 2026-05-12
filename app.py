import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from PIL import Image
from ultralytics import YOLO

# ---------------- LOAD MODEL ----------------
model = joblib.load("mask_detector_svm.pkl")
labels_map = ["with_mask", "without_mask", "incorrect_mask"]

# ---------------- LOAD YOLO FACE ----------------
yolo = YOLO("yolov8n-face.pt")  # make sure downloaded

# ---------------- UI ----------------
st.title("😷 Face Mask Detection")

mode = st.radio("Select Mode", ["Upload Image", "Real-Time Webcam"])

# ---------------- COMMON FUNCTION ----------------
def process_frame(img_bgr):
    results = yolo(img_bgr)[0]
    predictions = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        face = img_bgr[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_resized = cv2.resize(face, (128,128))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        features = hog(
            face_gray,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            feature_vector=True
        )

        pred = model.predict([features])[0]
        label = labels_map[pred]
        predictions.append(label)

        # Color
        if label == "with_mask":
            color = (0,255,0)
        elif label == "without_mask":
            color = (0,0,255)
        else:
            color = (0,255,255)

        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img_bgr, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return img_bgr, predictions

# ---------------- IMAGE MODE ----------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img = np.array(image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        result_img, preds = process_frame(img_bgr)

        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                 caption="Annotated Image")

        if preds:
            st.success(f"Detected: {len(preds)}")
            st.write("Prediction:", ", ".join(preds))
        else:
            st.warning("No face detected")

# ---------------- REAL-TIME MODE ----------------
# ---------------- REAL-TIME MODE ----------------
elif mode == "Real-Time Webcam":
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        result_frame, preds = process_frame(frame)

        FRAME_WINDOW.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))

    cap.release()