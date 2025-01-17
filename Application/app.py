from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import random

app = Flask(__name__)

# Load the trained model
model = load_model('model_50.h5')
# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Quiz questions (unchanged)
# Quiz questions (Machine Learning related)
questions = [
    # Easy Questions
    {"id": 1, "text": "What type of machine learning algorithm is used for predicting a continuous output?", "options": ["Classification", "Regression", "Clustering", "Dimensionality Reduction"], "answer": 1, "difficulty": "easy"},
    {"id": 2, "text": "Which of the following is an example of supervised learning?", "options": ["K-means clustering", "Linear regression", "Principal Component Analysis", "Autoencoders"], "answer": 1, "difficulty": "easy"},
    {"id": 3, "text": "What does the acronym 'SVM' stand for in machine learning?", "options": ["Statistical Variance Machine", "Support Vector Machine", "Structured Variable Model", "Supervised Vector Method"], "answer": 1, "difficulty": "easy"},
    {"id": 4, "text": "Which of the following is NOT a common activation function in neural networks?", "options": ["ReLU", "Sigmoid", "Tanh", "Gaussian"], "answer": 3, "difficulty": "easy"},
    {"id": 5, "text": "What is the primary goal of feature scaling in machine learning?", "options": ["Increase model complexity", "Reduce training time", "Normalize feature ranges", "Increase model accuracy"], "answer": 2, "difficulty": "easy"},

    # Medium Questions
    {"id": 6, "text": "Which algorithm is commonly used for dimensionality reduction?", "options": ["Random Forest", "K-Nearest Neighbors", "Principal Component Analysis", "Logistic Regression"], "answer": 2, "difficulty": "medium"},
    {"id": 7, "text": "What is the purpose of regularization in machine learning models?", "options": ["Increase model complexity", "Prevent overfitting", "Speed up training", "Improve feature selection"], "answer": 1, "difficulty": "medium"},
    {"id": 8, "text": "Which of the following is an ensemble learning method?", "options": ["K-means", "Support Vector Machine", "Random Forest", "Naive Bayes"], "answer": 2, "difficulty": "medium"},
    {"id": 9, "text": "What is the main difference between L1 and L2 regularization?", "options": ["L1 promotes sparsity, L2 doesn't", "L1 is faster to compute", "L2 is more effective", "There is no difference"], "answer": 0, "difficulty": "medium"},
    {"id": 10, "text": "Which optimization algorithm is commonly used in deep learning?", "options": ["Gradient Descent", "Simulated Annealing", "Genetic Algorithm", "A* Search"], "answer": 0, "difficulty": "medium"},

    # Hard Questions
    {"id": 11, "text": "What is the vanishing gradient problem in deep neural networks?", "options": ["Gradients become too large", "Gradients approach zero", "Gradients oscillate", "Gradients become complex numbers"], "answer": 1, "difficulty": "hard"},
    {"id": 12, "text": "Which of the following is a type of generative adversarial network (GAN)?", "options": ["LSTM", "BERT", "CycleGAN", "ResNet"], "answer": 2, "difficulty": "hard"},
    {"id": 13, "text": "What is the purpose of the attention mechanism in transformer models?", "options": ["To reduce model size", "To speed up training", "To focus on relevant parts of the input", "To prevent overfitting"], "answer": 2, "difficulty": "hard"},
    {"id": 14, "text": "Which of the following is NOT a common approach to handle imbalanced datasets?", "options": ["Oversampling", "Undersampling", "SMOTE", "Gradient boosting"], "answer": 3, "difficulty": "hard"},
    {"id": 15, "text": "What is the difference between batch normalization and layer normalization?", "options": ["Batch norm normalizes across the batch, layer norm across the features", "Batch norm is used in CNNs, layer norm in RNNs", "Batch norm is faster, layer norm is more accurate", "There is no difference"], "answer": 0, "difficulty": "hard"}
]

# User statistics
user_stats = {
    'total_questions': 0,
    'correct_answers': 0,
    'emotion_counts': {emotion: 0 for emotion in emotion_labels},
    'difficulty_counts': {'easy': 0, 'medium': 0, 'hard': 0},
    'emotion_difficulty_matrix': {emotion: {'easy': 0, 'medium': 0, 'hard': 0} for emotion in emotion_labels},
    'question_history': [], # Add this new list
    'answer_history': [] ,
    'current_question_id': None  # Add this new list
}

# Total emotion counts for the entire quiz duration
total_emotion_counts = {emotion: 0 for emotion in emotion_labels}

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (48, 48))
    normalized_frame = resized_frame / 255.0
    processed_frame = normalized_frame.reshape(1, 48, 48, 1)
    return processed_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    image_data = request.json['image']
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return jsonify({'error': 'No face detected'})
    
    # Process the first detected face
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Preprocess the face for emotion detection
    resized_face = cv2.resize(face_roi, (48, 48))
    normalized_face = resized_face / 255.0
    processed_face = normalized_face.reshape(1, 48, 48, 1)
    
    # Predict emotion
    prediction = model.predict(processed_face)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    
    total_emotion_counts[emotion] += 1
    
    return jsonify({'emotion': emotion, 'face': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}})


@app.route('/get_question', methods=['POST'])
def get_question():
    emotion = request.json['emotion']
    current_difficulty = request.json['currentDifficulty']
    consecutive_correct = request.json['consecutiveCorrect']
    last_emotion = request.json['lastEmotion']
    if emotion not in emotion_labels:
        emotion = 'Neutral' 
    if emotion == 'Happy':
        new_difficulty = 'hard'
    elif emotion in ['Fear', 'Disgust', 'Angry', 'Sad']:
        new_difficulty = 'easy'
    elif (emotion == 'Surprise' or emotion == 'Neutral') and emotion == last_emotion and consecutive_correct >= 2:
        new_difficulty = 'hard'
    else:
        new_difficulty = current_difficulty
    pro_random = 0.2
    if random.random() < pro_random:
        new_difficulty = 'hard'
    suitable_questions = [q for q in questions if q['difficulty'] == new_difficulty]
    if not suitable_questions:
        suitable_questions = questions  # Fallback to all questions if none match the difficulty
    
    question = random.choice(suitable_questions)
    question['difficulty'] = new_difficulty
    
    user_stats['total_questions'] += 1
    user_stats['emotion_counts'][emotion] += 1
    user_stats['difficulty_counts'][new_difficulty] += 1
    user_stats['emotion_difficulty_matrix'][emotion][new_difficulty] += 1
    # Update statistics
    user_stats['current_question_id'] = question['id']
    user_stats['question_history'].append({
        'question_number': user_stats['total_questions'],
        'question_id': question['id'],
        'emotion': emotion,
        'difficulty': new_difficulty,
        'correct': None  # Will be updated in check_answer
    })
    return jsonify(question)

#@app.route('/check_answer', methods=['POST'])
@app.route('/check_answer', methods=['POST'])
def check_answer():
    question_id = request.json['questionId']
    selected_option = request.json['selectedOption']
    current_difficulty = request.json['currentDifficulty']
    emotion = request.json['emotion']
    last_emotion = request.json['lastEmotion']
    consecutive_correct = request.json['consecutiveCorrect']
    
    # Find the question in the questions list
    question = next((q for q in questions if q['id'] == question_id), None)
    
    if not question:
        return jsonify({'error': 'Question not found'}), 404
    
    is_correct = selected_option == question['answer']

        # Update the last question's correctness
    if user_stats['question_history']:
        last_question = user_stats['question_history'][-1]
        if last_question['question_id'] == question_id:
            last_question['correct'] = is_correct
    
    
    if is_correct:
        consecutive_correct += 1
        user_stats['correct_answers'] += 1
        user_stats['answer_history'].append({
            'question_id': question_id,
            'correct': True
        })
    else:
        consecutive_correct = 0
        #user_stats['question_history'][-1]['correct'] = False
        user_stats['answer_history'].append({
            'question_id': question_id,
            'correct': False
        })
    
    # Difficulty adjustment logic (unchanged)
    if emotion == 'Happy':
        new_difficulty = 'hard'
    elif emotion in ['Fear', 'Disgust', 'Angry', 'Sad']:
        new_difficulty = 'easy'
    elif (emotion == 'Surprise' or emotion == 'Neutral') and emotion == last_emotion:
        if is_correct and consecutive_correct >= 1:
            new_difficulty = 'hard'
        elif not is_correct:
            new_difficulty = 'easy'
    elif not is_correct:
        new_difficulty = 'medium'
    else:
        new_difficulty = current_difficulty
    
    return jsonify({
        'isCorrect': is_correct,
        'newDifficulty': new_difficulty,
        'consecutiveCorrect': consecutive_correct
    })

@app.route('/get_stats', methods=['GET'])
def get_stats():
    return jsonify(user_stats)

@app.route('/get_total_emotions', methods=['GET'])
def get_total_emotions():
    return jsonify(total_emotion_counts)

if __name__ == '__main__':
    app.run(debug=True)