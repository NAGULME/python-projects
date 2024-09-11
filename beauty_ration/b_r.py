import cv2
import mediapipe as mp
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe Face Mesh for landmark detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)

# Golden Ratio (phi)
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # Approx 1.61803398875


# Define the neural network model for beauty score prediction
def build_beauty_score_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(3,)),  # 3 input features: face_ratio, eye_to_face_ratio, nose_to_face_ratio
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output beauty score (scaled between 0 and 1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


# Categorize the beauty score
def categorize_beauty_score(score):
    if score >= 0.85:
        return "Perfect(0.85)", (0, 255, 0)
    elif score >= 0.7:
        return "Good(0.7)", (0, 255, 255)
    elif score >= 0.5:
        return "Average(0.5)", (0, 165, 255)
    else:
        return "Bad", (0, 0, 255)


# Perform skin tone analysis (using the HSV color space)
def analyze_skin_tone(frame, landmarks):
    # Extract the forehead region for skin tone analysis
    x1, y1 = int(landmarks[10][0]), int(landmarks[10][1])  # Forehead point
    x2, y2 = int(landmarks[152][0]), int(landmarks[152][1])  # Chin point

    # Check if coordinates are within the frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    # Ensure the ROI is valid
    if x1 >= x2 or y1 >= y2:
        return "Invalid ROI", (0, 0, 0)

    # Define region of interest (ROI) for skin tone analysis
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Empty ROI", (0, 0, 0)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate the average color in the region
    avg_color = np.mean(hsv_roi, axis=(0, 1))

    hue = avg_color[0]  # Hue is used for determining skin tone
    if hue < 20:
        return "Light", (255, 220, 177)
    elif hue < 40:
        return "Medium", (209, 177, 122)
    else:
        return "Dark", (100, 67, 30)


# Calculate beauty ratios using 3D landmarks and the deep learning model
def calculate_beauty_ratios(landmarks, model):
    face_height = calculate_distance(landmarks[10], landmarks[152])  # Forehead to chin
    face_width = calculate_distance(landmarks[234], landmarks[454])  # Cheekbone to cheekbone
    eye_width = calculate_distance(landmarks[33], landmarks[263])    # Eye width
    nose_width = calculate_distance(landmarks[1], landmarks[5])      # Nose width

    face_ratio = face_height / face_width if face_width != 0 else 0
    eye_to_face_ratio = eye_width / face_width if face_width != 0 else 0
    nose_to_face_ratio = nose_width / face_width if face_width != 0 else 0

    # Predict beauty score using the deep learning model
    ratios = np.array([[face_ratio, eye_to_face_ratio, nose_to_face_ratio]])
    score = model.predict(ratios)

    return score[0][0], face_ratio, eye_to_face_ratio, nose_to_face_ratio


# Train the neural network model with example data
def train_model(model):
    # Example dataset (X are the ratios, y are the target beauty scores)
    X = np.array([
        [1.62, 0.98, 0.8],  # Near golden ratio examples
        [1.5, 1.1, 0.9],
        [1.55, 0.95, 0.88],
    ])
    y = np.array([0.9, 0.7, 0.8])  # Target beauty scores (scaled between 0 and 1)

    # Train the model
    model.fit(X, y, epochs=100)


# Draw suggestions and 3D improvements on the frame
def draw_golden_ratio_guidelines(frame, landmarks, face_ratio, eye_ratio, nose_ratio, score_category):
    cv2.putText(frame, f"Face Ratio: {face_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Eye Ratio: {eye_ratio:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Nose Ratio: {nose_ratio:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Provide suggestions based on ratios
    if abs(face_ratio - GOLDEN_RATIO) > 0.2:
        cv2.putText(frame, "Adjust face proportions for better symmetry", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    if abs(eye_ratio - GOLDEN_RATIO) > 0.2:
        cv2.putText(frame, "Eye proportions could be improved", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    if abs(nose_ratio - GOLDEN_RATIO) > 0.2:
        cv2.putText(frame, "Consider adjusting nose proportions", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    # Display the beauty score category (Perfect, Good, Average, Bad)
    cv2.putText(frame, f"Beauty Category: {score_category[0]}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                score_category[1], 2)


# Process each frame for 3D facial analysis, beauty ratio, and skin tone analysis
def process_frame(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks (3D coordinates)
            landmarks = [(face_landmarks.landmark[i].x * frame.shape[1],
                          face_landmarks.landmark[i].y * frame.shape[0],
                          face_landmarks.landmark[i].z * frame.shape[1]) for i in range(468)]

            # Calculate beauty ratios and score
            score, face_ratio, eye_ratio, nose_ratio = calculate_beauty_ratios(landmarks, model)
            score_category = categorize_beauty_score(score)

            # Display the beauty score
            cv2.putText(frame, f"Beauty Score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Perform skin tone analysis
            skin_tone, color = analyze_skin_tone(frame, landmarks)
            cv2.putText(frame, f"Skin Tone: {skin_tone}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw 3D facial landmarks and guidelines
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

            # Provide suggestions for improvements
            draw_golden_ratio_guidelines(frame, landmarks, face_ratio, eye_ratio, nose_ratio, score_category)

    return frame


# Initialize the model and train it
beauty_model = build_beauty_score_model()
train_model(beauty_model)

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame for 3D facial analysis, beauty score, and skin tone
    frame = process_frame(frame, beauty_model)

    # Show the output
    cv2.imshow("Enhanced Beauty Detector", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
