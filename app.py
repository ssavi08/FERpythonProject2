import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
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


# Define a custom VideoTransformer class to process frames from the webcam
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangle around faces and predict emotions
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


# Main function to run Streamlit app
def main():
    st.title("Facial Emotion Detection")
    st.write("OpenCV and Keras Streamlit App")

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)


if __name__ == "__main__":
    main()
