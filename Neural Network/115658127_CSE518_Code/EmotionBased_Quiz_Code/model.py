import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model_50.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the device camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera is ready. Press 'q' to quit.")

# Function to preprocess the frame for emotion detection
def preprocess_frame(face_region):
    resized_frame = cv2.resize(face_region, (48, 48))
    normalized_frame = resized_frame / 255.0
    processed_frame = normalized_frame.reshape(1, 48, 48, 1)
    return processed_frame

# Main loop for capturing and processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_region = gray_frame[y:y + h, x:x + w]

        # Preprocess the face for emotion detection
        processed_face = preprocess_frame(face_region)

        # Make a prediction
        prediction = model.predict(processed_face)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the emotion on the bounding box
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
