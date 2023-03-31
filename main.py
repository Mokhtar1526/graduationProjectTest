import tensorflow as tf
import cv2
import os
from os.path import exists
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from typing import List
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score
from keras import optimizers
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import HandModel, LoadModel

model = LoadModel.load()
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(result):
    lh = np.array(HandModel.HandModel(np.zeros(21 * 3)).feature_vector).flatten()
    # [[r.x, r.y, r.z] for r in result.left_hand_landmarks.landmark]
    # if result.left_hand_landmarks else
    rh = np.array(HandModel.HandModel([[r.x, r.y, r.z] for r in result.right_hand_landmarks.landmark]
                                      if result.right_hand_landmarks else np.zeros(21 * 3)).feature_vector).flatten()
    return np.concatenate([lh, rh])


colors = [(0, 0, 0), (255, 0, 0), (173, 216, 230), (240, 248, 255), (255, 255, 0), (34, 139, 34)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


# 1. New detection variables
sequence = []
sentence = []
threshold = 0.95
actions = np.array(["Shutdown", "Red", "Light_Blue", "Bright", "Yellow", "Green"])
print(actions)
cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        frame, results = mediapipe_detection(frame, hands)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-200:]

        if len(sequence) == 200:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            frame = prob_viz(res, actions, frame, colors)
            print(res[np.argmax(res)])
            if res[np.argmax(res)] > threshold:
                sentence.append(actions[np.argmax(res)])
                cv2.putText(frame, ' '.join(sentence), (60, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("image", frame)
        sentence.clear()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
