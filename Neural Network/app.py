from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import cv2
import numpy as np
import os
import random

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('final_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define mathematical questions
questions = {
    'easy': [
        {"question": "What is 2 + 2?", "options": ["3", "4", "5"], "answer": "4"},
        {"question": "What is 5 - 3?", "options": ["2", "3", "4"], "answer": "2"}
    ],
    'medium': [
        {"question": "What is 6 * 7?", "options": ["42", "40", "45"], "answer": "42"},
        {"question": "What is 15 / 3?", "options": ["5", "4", "6"], "answer": "5"}
    ],
    'difficult': [
        {"question": "What is 12^2?", "options": ["144", "121", "169"], "answer": "144"},
        {"question": "What is the square root of 81?", "options": ["7", "8", "9"], "answer": "9"}
    ]
}

def detect_emotion(image):
    np_array = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.05, 5)

# Detect faces
    faces = face_cascade.detectMultiScale(img, 1.05, 5)
    print(f"Number of faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        print(f"Face detected at (x: {x}, y: {y}, w: {w}, h: {h})")
        face_img = img[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (224, 224))
        reshaped = resized.reshape(1, 224, 224, 3) / 255
        predictions = model.predict(reshaped)
        print('Predictions:', predictions)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
        print('Predicted emotion:', emotions[max_index])
        return emotions[max_index]

    return 'neutral'

def get_question_based_on_difficulty(difficulty):
    if difficulty not in questions:
        difficulty = 'easy'
    question = random.choice(questions[difficulty])
    return question

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_route():
    image = request.files['image'].read()
    emotion = detect_emotion(image)
    return jsonify({'emotion': emotion})

@app.route('/get_question', methods=['POST'])
def get_question_route():
    emotion = request.json.get('emotion')
    difficulty_map = {
        'happy': 'easy',
        'sad': 'medium',
        'angry': 'difficult',
        'fear': 'medium',
        'neutral': 'easy',
        'surprise': 'medium',
        'disgust': 'difficult'
    }
    difficulty = difficulty_map.get(emotion, 'easy')
    question = get_question_based_on_difficulty(difficulty)
    return jsonify(question)

if __name__ == '__main__':
    app.run(debug=True)
