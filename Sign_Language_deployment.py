import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
import csv
import pandas as pd
import pyttsx3
import joblib
import time
import os

# # Definitions

# Paths to the landmarker model files
POSE_MODEL_PATH = 'pose_landmarker_heavy.task'
HAND_MODEL_PATH = 'hand_landmarker.task'

def mediapipe_detection(image, pose_landmarker, hand_landmarker):
    # Convert the image from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Process the image for pose and hands
    pose_results = pose_landmarker.detect(mp_image)
    hand_results = hand_landmarker.detect(mp_image)
    
    return pose_results, hand_results



# Manual definition of hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_styled_landmarks(image, hand_results, flash_active=False):
    if not hand_results.hand_landmarks:
        return
    
    # Determine color: Green if flashing, Blue/Black if not
    color = (0, 255, 0) if flash_active else (255, 0, 0)
    thickness = 3 if flash_active else 2
    
    for hand_landmarks in hand_results.hand_landmarks:
        # Convert landmarks to pixel coordinates
        h, w, _ = image.shape
        pixel_landmarks = []
        for landmark in hand_landmarks:
            px, py = int(landmark.x * w), int(landmark.y * h)
            pixel_landmarks.append((px, py))
            cv2.circle(image, (px, py), 3, color, -1)
            
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], color, thickness)


# # Make Detections


model_L = joblib.load('MP_model_head.pkl')



def sign_output(sentence, sentence_out):
    try:
        with open('multi_sign.csv', 'r') as multisign_file:
            sign_list_reader = csv.reader(multisign_file)
            for row in sign_list_reader:
                if len(sentence) >= 2:
                    if sentence[-1] == row[-1] and sentence[-2] == row[-2]:
                        sentence_out.append(row[0])
                        break
    except FileNotFoundError:
        print("multi_sign.csv not found")



def detect(vidsource):
    
    sentence = []
    sentence_out = []
    predictions = []
    last_sign_list = []
    one_sign_list = []
    
    threshold = 0.9
    pr = 3
    pTime = 0
    cTime = 0
    
    flash_counter = 0  # Counter for the 'flash green' effect
    
    # Loading complex signs mechanism
    try:
        with open('multi_sign.csv', 'r') as multisign_file:
            sign_list_reader = csv.reader(multisign_file)
            for row in sign_list_reader:
                if row:
                    last_sign_list.append(row[-1])
    except FileNotFoundError:
        print("multi_sign.csv not found")
    
    # Loading simple signs
    try:
        with open('single_sign.csv', 'r') as singlesign_file:
            singlesign_list_reader = csv.reader(singlesign_file)
            for row in singlesign_list_reader:
                if row:
                    one_sign_list.append(row[0])
    except FileNotFoundError:
        print("single_sign.csv not found")

    # MediaPipe Task Setup
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2
    )

    cap = cv2.VideoCapture(vidsource)
    
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            pose_results, hand_results = mediapipe_detection(frame, pose_landmarker, hand_landmarker)
            image = frame.copy()
            
            # Draw landmarks with flash effect
            draw_styled_landmarks(image, hand_results, flash_active=(flash_counter > 0))
            if flash_counter > 0:
                flash_counter -= 1

            # Extract landmarks
            # Hand landmarks extraction
            lh_row = list(np.zeros(21*3))
            rh_row = list(np.zeros(21*3))
            
            if hand_results.hand_landmarks:
                for idx, handedness in enumerate(hand_results.handedness):
                    label = handedness[0].category_name
                    landmarks = hand_results.hand_landmarks[idx]
                    flat_lms = list(np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten())
                    
                    if label == 'Left':
                        lh_row = flat_lms
                    elif label == 'Right':
                        rh_row = flat_lms

            # Pose (Head) extraction
            head = list(np.zeros(1*3))
            if pose_results.pose_landmarks:
                # Use nose landmark (index 0) as 'head'
                nose = pose_results.pose_landmarks[0][0]
                if nose.visibility > 0.8:
                    head = [nose.x, nose.y, nose.z]
            
            # Concatenate rows for model input
            row = lh_row + rh_row + head

            # Make Detections
            # Ensure the row has exactly 129 features
            if len(row) == 129:
                X = pd.DataFrame([row])
                sign_class = model_L.predict(X)[0]
                sign_prob = model_L.predict_proba(X)[0]

                # Sentence Logic
                max_prob = sign_prob[np.argmax(sign_prob)]
                if max_prob > threshold:
                    predictions.append(sign_class)

                    if len(predictions) >= pr and predictions[-pr:] == [sign_class]*pr:
                        if len(sentence) > 0:
                            if sign_class != sentence[-1]:
                                sentence.append(sign_class)
                                flash_counter = 5  # Flash green for 5 frames
                                
                                if sentence[-1] in last_sign_list:
                                    sign_output(sentence, sentence_out)
                                
                                if sentence[-1] in one_sign_list:
                                    sentence_out.append(sign_class)
                        else:
                            sentence.append(sign_class)
                            if sentence[-1] in one_sign_list:
                                    sentence_out.append(sign_class)

            if len(sentence) > 5:
                    sentence = sentence[-5:]
                    
            if len(sentence_out) > 6:
                    sentence_out = sentence_out[-6:]

            # UI Overlay
            cv2.rectangle(image, (0,0), (640, 40), (0,0,0), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.rectangle(image, (0, 45), (640, 85), (255,0,0), -1)
            cv2.putText(image, ' '.join(sentence_out), (3, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime

            cv2.putText(image, f"FPS: {int(fps)}", (10, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

            # Show to screen
            cv2.imshow('SignSema Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect(0)

