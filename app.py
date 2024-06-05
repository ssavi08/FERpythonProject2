import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

# Load the pre-trained model and Haar cascade classifier
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load model architecture from JSON file
with open('emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Reconstruct model from the JSON file
classifier = model_from_json(loaded_model_json)

# Load model weights from HDF5 file
classifier.load_weights('emotion_model.weights.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Function to process video frames and predict emotions
def predict_emotions(frame):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


# Function to display the video stream
def show_camera():
    cap = cv2.VideoCapture(0)
    video_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        frame = predict_emotions(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB")

        if st.session_state.stop_camera:
            break

    cap.release()


# Main function to set up Streamlit interface
def main():
    st.title("Predikcija osnovnih emocija prema izrazu lica")
    st.write("Projekt kolegija Primjenjeno strojno učenje")
    st.write("Made by: Savi & Hulak")

    if st.button("Pokreni kameru", key="start"):
        st.session_state.stop_camera = False
        show_camera()


if __name__ == "__main__":
    main()
