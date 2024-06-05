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


# Function to preprocess and predict emotions from an image
def predict_emotions(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to fit model input size
    roi_gray = cv2.resize(gray, (48, 48))

    # Normalize pixel values to [0, 1]
    roi = roi_gray.astype('float') / 255.0

    # Expand dimensions to match model input shape
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=3)

    # Predict emotion
    preds = emotion_model.predict(roi)[0]
    label = emotion_labels[np.argmax(preds)]

    return label


# Main function to run Streamlit app
def main():
    st.title("Facial Emotion Recognition")
    st.write("OpenCV and Keras Streamlit App")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image file
        image = cv2.imdecode(np.fromstring(uploaded_file.getvalue(), dtype=np.uint8), 1)

        # Predict emotions
        label = predict_emotions(image)

        # Display result
        st.write("Predicted emotion:", label)


if __name__ == "__main__":
    main()
