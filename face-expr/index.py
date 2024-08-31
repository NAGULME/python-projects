import cv2
import pygame
import numpy as np
from fer import FER

# Initialize pygame
pygame.init()

# Set the window size
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption('Face and Emotion Detection')

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    pygame.quit()
    exit()

# Initialize FER for emotion detection
detector = FER()

# Function to convert OpenCV image to Pygame surface
def cv2_to_pygame(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.transpose(img_rgb, (1, 0, 2)))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        continue

    # Detect faces and emotions
    faces = detector.detect_emotions(frame)

    for face in faces:
        if 'box' in face and 'emotions' in face:
            (x, y, w, h) = face['box']  # face bounding box
            emotion = face['emotions']  # emotion dictionary
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if emotion:
                emotion_label, emotion_score = max(emotion.items(), key=lambda item: item[1])
                cv2.putText(frame, f'{emotion_label}: {emotion_score:.2f}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            print(f"Unexpected face format: {face}")

    # Convert the frame to a Pygame surface
    frame_surface = cv2_to_pygame(frame)

    # Fill the background with white
    screen.fill(white)

    # Blit the frame surface onto the screen
    screen.blit(frame_surface, (0, 0))

    # Update the display
    pygame.display.update()

# Release resources and quit
cap.release()
pygame.quit()
