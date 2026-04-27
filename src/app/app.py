import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import os

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="BANTAI", layout="wide")

st.title(" BANTAI - Drowsiness Detection")
st.markdown("Real-time eye monitoring system using Deep Learning")

col1, col2 = st.columns([2, 1])

# Sidebar controls
st.sidebar.title("Controls")
run = st.sidebar.button("Start Camera")
stop = st.sidebar.button("Stop")

threshold = st.sidebar.slider("Closed Threshold", 0.1, 0.6, 0.25)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("models/model.h5")
IMG_SIZE = (224, 224)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FRAME_WINDOW = col1.image([])
status_box = col2.empty()
confidence_box = col2.empty()

closed_start = None
cap = cv2.VideoCapture(0)

# ---------------- LOOP ----------------
while run and not stop:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    label = "No Face"
    color = (0, 255, 255)
    confidence = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = frame[y:y+h, x:x+w]
        eye_region = face[int(h*0.2):int(h*0.5), int(w*0.2):int(w*0.8)]

        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_gray = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(eye_gray, IMG_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)[0][0]
        confidence = float(prediction)

        if prediction < threshold:
            if closed_start is None:
                closed_start = time.time()

            elapsed = time.time() - closed_start

            if elapsed > 2:
                label = "😴 Sleeping"
                color = (0, 0, 255)
                os.system("afplay sound/fahhhhh.mp3 &")
            else:
                label = "😪 Drowsy"
                color = (0, 255, 255)
        else:
            closed_start = None
            label = "👀 Awake"
            color = (0, 255, 0)

        break

    # Draw label
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    # Convert frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # ---------------- UI DISPLAY ----------------
    if "Awake" in label:
        status_box.success(label)
    elif "Drowsy" in label:
        status_box.warning(label)
    elif "Sleeping" in label:
        status_box.error(label)
    else:
        status_box.info(label)

    confidence_box.metric("Model Confidence", f"{confidence:.2f}")

cap.release()