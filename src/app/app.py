import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import altair as alt
import cv2
import time
import os
from datetime import datetime


# FEEDBACK HELPER FUNCTION
def save_feedback(
    start_time,
    duration,
    drowsy_episodes,
    sleep_episodes,
    feedback_rating
):
    os.makedirs("logs", exist_ok=True)
    file_path = "logs/feedback.csv"

    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "start_time": start_time,
        "duration_minutes": round(duration / 60, 2),
        "drowsy_episodes": drowsy_episodes,
        "sleep_episodes": sleep_episodes,
        "feedback_rating": feedback_rating
    }

    df_new = pd.DataFrame([data])
    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(file_path, index=False)


# PAGE PREREQUISITES
st.set_page_config(page_title="BANTAI", layout="wide")
st.title("BANTAI - Drowsiness Detection")
st.markdown("Real-time eye monitoring system using Deep Learning")
col1, col2 = st.columns([2, 1])


# CURRENT STREAMLIT RUN INSTANCE
STATE = st.session_state


# PAUSED SESSION MESSAGE
pause_box = st.empty()
if STATE.get("camera_paused", False):
    pause_box.warning("Session is currently paused. Click **'Resume Session'** to continue.")
else:
    pause_box.empty()


# SESSION SUMMARY FEATURES
if 'start_time' not in STATE:
    STATE.start_time = None               # Local start time
if 'duration' not in STATE:
    STATE.duration = None                 # Session start time
if 'session_length' not in STATE:
    STATE.session_length = 0.0            # Session total time
if 'drowsy_episodes' not in STATE:
    STATE.drowsy_episodes = 0             # Number of 'drowsy' episodes
if 'sleep_episodes' not in STATE:
    STATE.sleep_episodes = 0              # Number of 'sleeping' episodes
if 'alertness_timeline' not in STATE:
    STATE.alertness_timeline = []         # List of episodes
if 'prediction_history' not in STATE:
    STATE.prediction_history = []


# TIME INTERVAL HELPERS
if 'closed_start' not in STATE:
    STATE.closed_start = None   
if 'last_drowsy_time' not in STATE:
    STATE.last_drowsy_time = 0
if 'last_sleep_time' not in STATE:
    STATE.last_sleep_time = 0
if 'last_alarm_time' not in STATE:
    STATE.last_alarm_time = 0


# START/PAUSE/RESUME/STOP HELPERS
if 'running_time' not in STATE:
    STATE.running_time = 0.0               # Unpaused total session time
if 'camera_paused' not in STATE:
    STATE.camera_paused = False
if 'camera_running' not in STATE:
    STATE.camera_running = False
if 'session_stopped' not in STATE:
    if 'feedback_submitted' not in STATE:
        STATE.feedback_submitted = False
    STATE.session_stopped = False


# BUTTON LABEL HELPERS
if STATE.camera_running:
    button_label = "Pause Session"
elif STATE.camera_paused:
    button_label = "Resume Session"
else:
    button_label = "Start Session"


# SIDEBAR CONTROLS
st.sidebar.title("Controls")

if st.sidebar.button(button_label):
    if button_label == "Start Session":
        STATE.camera_running = True
        STATE.camera_paused = False
        STATE.session_stopped = False
        STATE.duration = None

    elif button_label == "Pause Session":
        STATE.camera_running = False
        STATE.camera_paused = True
        STATE.running_time += time.time() - STATE.duration

    elif button_label == "Resume Session":
        STATE.camera_running = True
        STATE.camera_paused = False
        STATE.duration = time.time()

    st.rerun()

elif st.sidebar.button("Stop"):
    # Freeze final session length
    if STATE.camera_running and STATE.duration is not None:
        STATE.session_length = STATE.running_time + time.time() - STATE.duration
    elif STATE.camera_paused:
        STATE.session_length = STATE.running_time

    STATE.camera_running = False
    STATE.camera_paused = False
    STATE.session_stopped = True
    
    st.rerun()


# USER-CONFIGURABLE PREFERENCES
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Preferences")
threshold = st.sidebar.slider("Eye Closed Threshold", 0.1, 1.0, 0.25)
alarm_interval = st.sidebar.slider("Alarm Repeat Interval (seconds)", 2.0, 10.0, 3.0, 0.5)


# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/model.h5", compile=False)

model = load_model()
IMG_SIZE = (224, 224)
DROWSY_COOLDOWN = 5
SLEEP_COOLDOWN = 8


# LOAD CAMERA
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FRAME_WINDOW = col1.image([])
status_box = col2.empty()
confidence_box = col2.empty()
speed_box = col2.empty()
cap = cv2.VideoCapture(0)


# NEW SESSION REINITIALIZATION
if STATE.camera_running and STATE.duration is None:
    STATE.start_time = datetime.now().strftime("%I:%M %p")
    STATE.duration = time.time()
    STATE.session_length = 0
    STATE.drowsy_episodes = 0
    STATE.sleep_episodes = 0
    STATE.alertness_timeline = []

    STATE.closed_start = None
    STATE.last_drowsy_time = 0
    STATE.last_sleep_time = 0
    STATE.last_alarm_time = 0

    STATE.running_time = 0.0
    STATE.camera_paused = False
    STATE.session_stopped = False
    STATE.feedback_submitted = False


# MAIN DROWSINESS DETECTION LOOP
while STATE.camera_running:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    label = "No Face"
    color = (0, 255, 255)
    confidence = 0
    inference_time = 0

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # crop both eyes
        eye_region = face[
            int(h * 0.18):int(h * 0.50),
            int(w * 0.15):int(w * 0.85)
        ]

        # Skip empty frames
        if eye_region.size == 0:
            continue

        # Keep RGB (better for glasses)
        eye_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)

        # Light blur to reduce glasses glare/noise
        eye_rgb = cv2.GaussianBlur(eye_rgb, (3, 3), 0)

        # Resize
        img = cv2.resize(eye_rgb, IMG_SIZE)

        # Normalize
        img = img.astype("float32") / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        start_inference = time.time()

        # Predict
        prediction = model.predict(img, verbose=0)[0][0]

        end_inference = time.time()

        inference_time = end_inference - start_inference    

        # Store history
        STATE.prediction_history.append(prediction)

        # Only keep last 10 predictions
        if len(STATE.prediction_history) > 10:
            STATE.prediction_history.pop(0)

        # smooth predictions (average yung recent predictions)
        prediction = np.mean(STATE.prediction_history)

        confidence = float(prediction)


        # if len(STATE.prediction_history) > 10:
        #     STATE.prediction_history.pop(0)

        # prediction = np.mean(STATE.prediction_history)

        confidence = float(prediction)
        print(prediction)

        # Identify user state
        if prediction < threshold:
            if STATE.closed_start is None:
                STATE.closed_start = time.time()
            elapsed = time.time() - STATE.closed_start

            if elapsed > 3:
                label = "Sleeping"
                color = (0, 0, 255)

                if time.time() - STATE.last_alarm_time > alarm_interval:
                    os.system("afplay sound/fah_edited.mp3 &")
                    STATE.last_alarm_time = time.time()
            elif elapsed > 1:
                label = "Drowsy"
                color = (0, 255, 255)
            else:
                label = "Awake"
                color = (0, 255, 0)
        else:
            STATE.closed_start = None
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
    speed_box.metric(
        "Inference Time",
        f"{inference_time*1000:.2f} ms"
    )

    # Store detected states in alertness_timeline
    if STATE.duration is not None:
        # Get second when frame was processed
        curr = STATE.running_time + time.time() - STATE.duration
        # Scale categories numerically from 0 to 1
        if "No Face" in label: val = np.nan
        else: val = 0.0 if "Sleeping" in label else (0.5 if "Drowsy" in label else 1.0)

        current_time = time.time()

        # New sleep episode
        if val == 0.0 and STATE.closed_start is not None:
            if current_time - STATE.last_sleep_time > SLEEP_COOLDOWN:
                STATE.sleep_episodes += 1
                STATE.last_sleep_time = current_time

        # New drowsy episode
        elif val == 0.5:
            if current_time - STATE.last_drowsy_time > DROWSY_COOLDOWN:
                STATE.drowsy_episodes += 1
                STATE.last_drowsy_time = current_time

        # Set time as the x-value and alertness level as the y-value
        STATE.alertness_timeline.append({
            "Clock Time": datetime.now().strftime("%I:%M:%S %p"),
            "Time (seconds)": curr, 
            "Alertness Level": val
        })

cap.release()


# SESSION SUMMARY (AFTER 'STOP')
if STATE.session_stopped and STATE.duration is not None:
    st.markdown("---")
    st.header("Session Summary")
    
    # Calculate total time (in minutes and seconds)
    mins, secs = divmod(int(STATE.session_length), 60)
    
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
        chart = alt.Chart(df).mark_area(color='#99ccff').encode(
            x=alt.X("Time (seconds):Q"),
            y=alt.Y("Alertness Level:Q"),
            tooltip=[
                alt.Tooltip("Clock Time:N"),
                alt.Tooltip("Time (seconds):Q", format=".1f"), 
                alt.Tooltip("Alertness Level:Q") 
            ]
        ).configure_axisX(grid=False).configure_view(strokeOpacity=0)
        st.altair_chart(chart, use_container_width=True)


        # PERSONALIZED RECOMMENDATIONS
        negative_episodes = df[df["Alertness Level"] < 1.0]     # Isolate drowsy/sleeping episodes

        # Scenario 1: Short session
        if STATE.session_length < 30:
            st.info("Session too short for meaningful behavioral recommendations.")

        # Scenario 2: Fully awake
        elif STATE.sleep_episodes + STATE.drowsy_episodes == 0:
            st.success(f"#### 💡 Personal Recommendations\n You managed to stay fully awake throughout the session! Continue your study habits, and don't forget to get some rest after a job well done.")
        
        elif not negative_episodes.empty:
            # Get first instance of drowsy/sleeping
            episode_incidence_min, episode_incidence_sec = divmod(int(negative_episodes.index[0]), 60)

            # String helper
            if episode_incidence_min > 0 and episode_incidence_sec > 0:
                minute_label = "minute" if episode_incidence_min == 1 else "minutes"
                second_label = "second" if episode_incidence_sec == 1 else "seconds"
                focus_time = (
                    f"{episode_incidence_min} {minute_label} "
                    f"and {episode_incidence_sec} {second_label}"
                )
            elif episode_incidence_min > 0:
                minute_label = "minute" if episode_incidence_min == 1 else "minutes"
                focus_time = f"{episode_incidence_min} {minute_label}"
            else:
                second_label = "second" if episode_incidence_sec == 1 else "seconds"
                focus_time = f"{episode_incidence_sec} {second_label}"

            # Scenario 3: Immediate fatigue
            if episode_incidence_min < 15:
                if episode_incidence_min == 0 and episode_incidence_sec < 10:
                    st.warning(
                        "#### 💡 Personal Recommendations\n"
                        "Early signs of fatigue were detected shortly after the session began. "
                        "This may indicate tiredness, eye strain, or temporary detection noise. "
                        "Try adjusting your lighting, posture, or taking a short rest before studying again."
                    )
                else:
                    st.warning(
                        f"#### 💡 Personal Recommendations\n"
                        f"You began experiencing fatigue after {focus_time} of studying. "
                        f"Consider taking short breaks and ensuring you are well-rested before long study sessions."
                    )

            # Scenario 4: Fatigue pattern recognition
            else:
                recommended_break = max(25, int(episode_incidence_min * 0.75))
                st.info(
                    f"#### 💡 Personal Recommendations\n"
                    f"You consistently stayed focused for {focus_time} before feeling drowsy. "
                    f"Based on your alertness trend, taking a short break every "
                    f"{recommended_break} minutes may help maintain focus and reduce fatigue."
                )


        # SESSION FEEDBACK
        st.markdown("---")
        st.subheader("Session Feedback")
        st.write("How accurate was BANTAI during this session?")

        feedback = st.radio(
            "Select a rating:",
            [
                "⭐ Very Accurate",
                "🙂 Accurate",
                "😐 Neutral",
                "😕 Inaccurate",
                "❌ Very Inaccurate"
            ]
        )

        if not STATE.feedback_submitted and st.button("Submit Feedback"):
            save_feedback(
                STATE.start_time,
                STATE.session_length,
                STATE.drowsy_episodes,
                STATE.sleep_episodes,
                feedback
            )
            STATE.feedback_submitted = True
            st.success("Feedback submitted successfully!")
            st.balloons()
