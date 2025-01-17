from graphviz import Digraph

# Create the graph
dot = Digraph(comment='Emotion-Aware E-Learning Platform')

# Backend
dot.node('A', 'Flask Backend\n(Python)')
dot.node('B', 'Emotion Detection Model\n(CNN)')
dot.node('C', 'Adaptive Quiz System')
dot.node('D', 'Data Management')
dot.node('E', 'User Stats')

# Frontend
dot.node('F', 'Frontend\n(HTML, CSS, JS)')
dot.node('G', 'Webcam Feed')
dot.node('H', 'Emotion Feedback')

# Communication & Visualization
dot.node('I', 'AJAX Requests\n(Real-Time Data Exchange)')
dot.node('J', 'Chart.js\n(Performance Visualization)')
dot.node('K', 'Real-Time Feedback\n(Emotion & Difficulty)')

# Define edges between nodes to represent the flow
dot.edge('F', 'G', label='Webcam Feed')
dot.edge('G', 'B', label='Face Detection')
dot.edge('B', 'H', label='Emotion Prediction')
dot.edge('F', 'I', label='AJAX Requests')
dot.edge('I', 'A', label='Send Emotion Data')
dot.edge('A', 'C', label='Select Quiz Questions')
dot.edge('C', 'A', label='Quiz Feedback')
dot.edge('A', 'E', label='Track User Stats')
dot.edge('E', 'A', label='Update Stats')
dot.edge('A', 'D', label='Data Management')
dot.edge('D', 'A', label='Emotion-Difficulty Matrix')

dot.edge('A', 'F', label='Dynamic Content Delivery\n(Quiz and Feedback)')
dot.edge('F', 'J', label='Charts for Real-Time Stats')
dot.edge('F', 'K', label='Emotion-Based Difficulty Adjustments')

# Add overall flow of system
dot.node('L', 'User')
dot.edge('L', 'F', label='Interact with the System')
dot.edge('F', 'A', label='Send Data for Processing')

# Render the diagram to a PNG file
dot.render('/mnt/data/emotion_aware_learning_platform', format='png')

# Display the file path
'/mnt/data/emotion_aware_learning_platform.png'
