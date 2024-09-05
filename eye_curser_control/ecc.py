#pip3 install mediapipe
#pip3 install pyautogui
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe face mesh detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define eye landmark indices for left and right eyes
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 153, 154, 155]
RIGHT_EYE_INDICES = [263, 362, 387, 386, 385, 373, 380, 374, 380]


# Helper function to calculate the eye center
def calculate_eye_center(landmarks, frame_width, frame_height, eye_indices):
    eye_points = np.array(
        [(int(landmarks[i].x * frame_width), int(landmarks[i].y * frame_height)) for i in eye_indices])
    return np.mean(eye_points, axis=0).astype(int)


# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and process with Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            frame_height, frame_width = frame.shape[:2]

            # Calculate eye centers
            left_eye_center = calculate_eye_center(landmarks.landmark, frame_width, frame_height, LEFT_EYE_INDICES)
            right_eye_center = calculate_eye_center(landmarks.landmark, frame_width, frame_height, RIGHT_EYE_INDICES)

            # Average eye center and map to screen coordinates
            eye_center = (left_eye_center + right_eye_center) // 2
            screen_x = np.interp(eye_center[0], [0, frame_width], [0, pyautogui.size()[0]])
            screen_y = np.interp(eye_center[1], [0, frame_height], [0, pyautogui.size()[1]])

            # Move the cursor
            pyautogui.moveTo(screen_x, screen_y)

            # Debug: Draw eye centers on frame
            cv2.circle(frame, tuple(left_eye_center), 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(right_eye_center), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Eye Tracking", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
