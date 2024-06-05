import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

# Load model architecture from JSON file
with open('emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Reconstruct model from the JSON file
emotion_model = model_from_json(loaded_model_json)

# Load model weights from HDF5 file
emotion_model.load_weights('emotion_model.weights.h5')

# Define a dictionary to map emotion labels to their corresponding indices
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to predict emotions from a frame
def predict_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        preds = emotion_model.predict(roi)[0]
        label = emotion_labels[np.argmax(preds)]

        # Draw rectangle around face and label emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame


# Function to capture video from webcam and display emotion predictions
def main():
    st.title("Facial Emotion Detection")
    st.write("OpenCV and Keras Streamlit App")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        frame = predict_emotions(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

        if st.button("Stop"):
            break

    cap.release()


if __name__ == "__main__":
    main()
