import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import time
import os
from datetime import datetime


st.set_page_config(page_title="BANTAI", layout="wide")

st.title("BANTAI - Drowsiness Detection")
st.markdown("Real-time eye monitoring system using Deep Learning")

col1, col2 = st.columns([2, 1])

# Sidebar controls
st.sidebar.title("Controls")
run = st.sidebar.button("Start Camera")
stop = st.sidebar.button("Stop")

threshold = st.sidebar.slider("Closed Threshold", 0.1, 0.6, 0.25)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/model.h5", compile=False)

model = load_model()
IMG_SIZE = (224, 224)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FRAME_WINDOW = col1.image([])
status_box = col2.empty()
confidence_box = col2.empty()

closed_start = None
cap = cv2.VideoCapture(0)

STATE = st.session_state

# Key features of session summary
if 'duration' not in STATE: STATE.duration = None                           # Session duration
if 'drowsy_episodes' not in STATE: STATE.drowsy_episodes = 0                # Number of 'drowsy' episodes
if 'sleep_episodes' not in STATE: STATE.sleep_episodes = 0                  # Number of 'sleeping' episodes
if 'drowsy_flag' not in STATE: STATE.drowsy_flag = False                    # Prevents duplicated 'drowsy' episodes
if 'sleep_flag' not in STATE: STATE.sleep_flag = False                      # Prevents duplicated 'sleeping' episodes
if 'alertness_timeline' not in STATE: STATE.alertness_timeline = []         # List of episodes
if 'start_time' not in STATE: STATE.start_time = None                       # Local start time

# Start new session on new running instance
if run:
    STATE.duration = time.time()
    STATE.drowsy_episodes = 0
    STATE.sleep_episodes = 0
    STATE.drowsy_flag = False
    STATE.sleep_flag = False
    STATE.alertness_timeline = []
    STATE.start_time = datetime.now().strftime("%I:%M %p")

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
                label = "Sleeping"
                color = (0, 0, 255)
                os.system("afplay sound/fahhhhh.mp3 &")
            elif elapsed > 1:
                label = "Drowsy"
                color = (0, 255, 255)
            else:
                label = "Awake"
                color = (0, 255, 0)
        else:
            closed_start = None
            label = "Awake"
            color = (0, 255, 0)

        break

    # Draw label
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    # Convert frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)


    if "Awake" in label:
        status_box.success(label)
    elif "Drowsy" in label:
        status_box.warning(label)
    elif "Sleeping" in label:
        status_box.error(label)
    else:
        status_box.info(label)

    confidence_box.metric("Model Confidence", f"{confidence:.2f}")

    if STATE.duration is not None:
        # Get second when frame was processed
        curr = time.time() - STATE.duration
        # Scale categories numerically from 0 to 1
        if "No Face" in label: val = np.nan
        else: val = 0.0 if "Sleeping" in label else (0.5 if "Drowsy" in label else 1.0)

        # New sleep episode
        if val == 0.0 and not STATE.sleep_flag:
            STATE.sleep_episodes += 1
            STATE.sleep_flag = True

        # New drowsy episode
        if val == 0.5 and not STATE.drowsy_flag:
            STATE.drowsy_episodes += 1
            STATE.drowsy_flag = True

        if val != 0.0: STATE.sleep_flag = False
        if val != 0.5: STATE.drowsy_flag = False

        # Set time as the x-value and alertness level as the y-value
        STATE.alertness_timeline.append({"Time (seconds)": curr, "Alertness Level": val})

cap.release()

# Session summary upon 'Stop'
if stop and STATE.duration is not None:
    st.markdown("---")
    st.header("Session Summary")
    
    # Calculate total time (in minutes and seconds)
    mins, secs = divmod(int(time.time() - STATE.duration), 60)
    
    # Set up data columns
    a, b, c, d = st.columns(4)
    a.metric("Start Time", STATE.start_time)
    b.metric("Total Session Duration", f'{mins}m {secs}s')
    c.metric("Drowsy Episodes Detected", STATE.drowsy_episodes)
    d.metric("Sleep Episodes Detected", STATE.sleep_episodes)
    
    # Alertness timeline
    if len(STATE.alertness_timeline) > 0:
        st.subheader("Alertness Timeline")
        st.markdown("<div style='text-align: right; color: gray; font-size: 1rem;'>"
                    "0.0: Sleeping | 0.5: Drowsy | 1.0: Awake</div>", 
                    unsafe_allow_html=True)
        df = pd.DataFrame(STATE.alertness_timeline)
        df.set_index("Time (seconds)", inplace=True)
        st.area_chart(df, x_label="Time (seconds)", y_label="Alertness Level")

        # Personalized recommendations
        negative_episodes = df[df["Alertness Level"] < 1.0]     # Isolate drowsy/sleeping episodes
        # Scenario 1: Fully awake
        if STATE.sleep_episodes + STATE.drowsy_episodes == 0: st.success(f"#### 💡 Personal Recommendations\n You managed to stay fully awake throughout the session! Continue your study habits, and don't forget to get some rest after a job well done.")
        elif not negative_episodes.empty:
            # Get first instance of drowsy/sleeping
            episode_incidence_min, episode_incidence_sec = divmod(int(negative_episodes.index[0]), 60)

            # String helper
            if episode_incidence_min > 0 and episode_incidence_sec > 0: focus_time = f"{episode_incidence_min} minutes and {episode_incidence_sec} seconds"
            elif episode_incidence_min > 0: focus_time = f"{episode_incidence_min} minutes"
            else: focus_time = f"{episode_incidence_sec} seconds"

            # Scenario 2: Immediate fatigue
            if episode_incidence_min < 10: st.warning(f"#### 💡 Personal Recommendations\n You immediately experienced fatigue just {focus_time} into the session. Consider resting before studying again.")
            # Scenario 3: Fatigue pattern recognition
            else: st.info(f"#### 💡 Personal Recommendations\n You consistently stayed focused for {focus_time} before feeling drowsy. Consider taking a break every {max(10, episode_incidence_min - 10)} minutes for optimal performance.")