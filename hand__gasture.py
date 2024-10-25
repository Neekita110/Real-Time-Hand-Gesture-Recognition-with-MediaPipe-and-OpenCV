import os
import cv2
import mediapipe as mp
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


# Gesture recognition function
def recognize_gesture(hand_landmarks):
    thumb_is_open = hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y
    index_finger_is_open = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_finger_is_open = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    ring_finger_is_open = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    pinky_is_open = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y

    if thumb_is_open and not index_finger_is_open and not middle_finger_is_open and not ring_finger_is_open and not pinky_is_open:
        return "Thumbs Up (All OK)"
    elif not thumb_is_open and index_finger_is_open and not middle_finger_is_open and not ring_finger_is_open and not pinky_is_open:
        return "Cross (Not OK)"
    elif thumb_is_open and index_finger_is_open and middle_finger_is_open and ring_finger_is_open and pinky_is_open:
        return "Palm (Stop)"
    elif index_finger_is_open and middle_finger_is_open and ring_finger_is_open and pinky_is_open:
        return "Wave (Help)"
    else:
        return "Unknown Gesture"


# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks)
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing ESC
        break

cap.release()
cv2.destroyAllWindows()
