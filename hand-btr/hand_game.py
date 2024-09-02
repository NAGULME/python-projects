import cv2  # Import OpenCV for video capturing and processing
import mediapipe as mp  # Import MediaPipe for hand gesture detection
import random  # Import random module to randomly select AI's move
from gtts import gTTS  # Import gTTS for text-to-speech conversion
import os  # Import os to play the sound file

# Initialize MediaPipe Hands for hand tracking and gesture detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)  # Process only one hand with 70% confidence
mp_draw = mp.solutions.drawing_utils  # Utility to draw hand landmarks on the image

# Initialize OpenCV Video Capture to capture video from the webcam
cap = cv2.VideoCapture(0)


# Function to classify the gesture as Rock, Paper, or Scissors based on hand landmarks
def classify_gesture(landmarks, h, w):
    # Extract y-coordinates of fingertips (scaled to image height)
    thumb_tip_y = landmarks[mp_hands.HandLandmark.THUMB_TIP].y * h
    index_tip_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
    middle_tip_y = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * h
    ring_tip_y = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * h
    pinky_tip_y = landmarks[mp_hands.HandLandmark.PINKY_TIP].y * h

    # Determine gesture based on the relative positions of fingertips
    if thumb_tip_y > index_tip_y and middle_tip_y > index_tip_y and ring_tip_y > index_tip_y and pinky_tip_y > index_tip_y:
        return 'Rock'  # All fingers are down
    elif thumb_tip_y < index_tip_y and middle_tip_y < index_tip_y and ring_tip_y < index_tip_y and pinky_tip_y < index_tip_y:
        return 'Paper'  # All fingers are up
    else:
        return 'Scissors'  # Index and middle fingers are up


# Function to determine the winner between player and AI
def determine_winner(player_move, ai_move):
    if player_move == ai_move:
        return "Tie"  # Both player and AI chose the same move
    elif (player_move == "Rock" and ai_move == "Scissors") or \
            (player_move == "Paper" and ai_move == "Rock") or \
            (player_move == "Scissors" and ai_move == "Paper"):
        return "Player Wins"  # Player's move beats AI's move
    else:
        return "MEERA WINS"  # AI's move beats player's move


# Function to announce the winner using gTTS and play the audio
def announce_winner(winner):
    if winner == "Player Wins":
        text = "Congratulations! You are the winner!"
    elif winner == "MEERA WINS":
        text = "MEERA WINS this round!"
    else:
        text = "It's a tie!"

    tts = gTTS(text=text, lang='en')  # Convert text to speech
    tts.save("winner.mp3")  # Save the speech as an MP3 file
    os.system("afplay winner.mp3")  # Play the MP3 file (use "mpg321" for Linux or "start" for Windows)


# Main loop for capturing video and processing hand gestures
while True:
    success, img = cap.read()  # Capture a frame from the webcam
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB (required for MediaPipe)
    results = hands.process(img_rgb)  # Process the image and detect hand landmarks

    player_move = None  # Initialize player_move as None
    ai_move = random.choice(['Rock', 'Paper', 'Scissors'])  # Randomly select AI's move

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw landmarks on the image
            h, w, _ = img.shape  # Get the height and width of the image
            player_move = classify_gesture(hand_landmarks.landmark, h, w)  # Classify the gesture

    # If a player move is detected, determine and display the result
    if player_move:
        winner = determine_winner(player_move, ai_move)  # Determine the winner
        cv2.putText(img, f"Player: {player_move}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                    2)  # Display player's move
        cv2.putText(img, f"AI: {ai_move}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Display AI's move
        cv2.putText(img, f"{winner}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)  # Display the winner
        announce_winner(winner)  # Announce the winner via speech

    # If no player move is detected, prompt the player to show a gesture
    else:
        cv2.putText(img, "Show Rock, Paper, or Scissors", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Rock-Paper-Scissors", img)  # Display the image with the overlays

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
