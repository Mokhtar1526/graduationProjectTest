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
import HandModel


def load():
    # ACTION 12 TEST
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, dropout=0.2, input_shape=(200, 882)))
    model.add(LSTM(64, return_sequences=True, dropout=0.2))
    model.add(LSTM(32, dropout=0.2))
    model.add(Dense(6, activation='softmax'))
    adam = optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('C:/Users/mikha/PycharmProjects/graduationProjectTest/action12.h5')
    return model
