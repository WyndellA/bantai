import os
import cv2
import numpy as np
import tensorflow as tf
import time   

# Load model
model = tf.keras.models.load_model("models/model.h5")

IMG_SIZE = (224, 224)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

closed_start = None 
alarm_played = False
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    label = "No Face Detected"
    color = (0, 255, 255)

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = frame[y:y+h, x:x+w]

        # eye region crop
        eye_region = face[int(h*0.2):int(h*0.5), int(w*0.2):int(w*0.8)]

        # Convert to grayscale
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

        # Convert back to 3 channels (model expects RGB)
        eye_gray = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2BGR)

        # Resize + normalize
        img = cv2.resize(eye_gray, IMG_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img, verbose=0)[0][0]
        print("Prediction:", round(prediction, 3))

        # Time-based logic
        if prediction < 0.25:
            if closed_start is None:
                closed_start = time.time()

            elapsed = time.time() - closed_start

            if elapsed > 2:
                label = "Sleeping"
                color = (0, 0, 255)
                os.system("afplay sound/fahhhhh.mp3 &")

            else:
                label = "Drowsy"
                color = (0, 255, 255)
        else:
            closed_start = None
            alarm_played = False
            label = "Awake 👀"
            color = (0, 255, 0)

        # Show eye region 
        cv2.imshow("Eye Region", eye_gray)

        break  # process only first face

    # Display result
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    cv2.imshow("BANTAI - Eye Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()