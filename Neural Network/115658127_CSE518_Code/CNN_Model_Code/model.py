import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
print(tf.__version__)

# Load model
model = load_model("final_model.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Color dictionary for drawing
GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Initialize video capture
cap = cv2.VideoCapture(0)
output = []
i = 0

while i <= 50:
    ret, img = cap.read()
    
    if not ret:
        break

    # Flip the image horizontally to create a mirror effect
    img = cv2.flip(img, 1)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(img, 1.05, 5)
    
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        
        # Preprocess the face image
        resized = cv2.resize(face_img, (224, 224))
        reshaped = resized.reshape(1, 224, 224, 3) / 255
        predictions = model.predict(reshaped)
        
        # Find the index of the maximum prediction
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
        predicted_emotion = emotions[max_index]
        output.append(predicted_emotion)
        
        # Draw rectangle and label
        cv2.rectangle(img, (x, y), (x+w, y+h), GR_dict[1], 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), GR_dict[1], -1)
        cv2.putText(img, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    i += 1

    # Display the image with predictions
    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' key to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print output and compute the most frequent emotion
print(output)
most_common_emotion = Counter(output).most_common(1)[0][0]
print("Most frequent emotion:", most_common_emotion)
