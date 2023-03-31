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
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn import metrics


class HandModel(object):
    """
    Params
        landmarks: List of positions
    Args
        connections: List of tuples containing the ids of the two landmarks representing a connection
        feature_vector: list of length 21*21=441 containing the angles between all connections
    """

    def __init__(self, landmarks: List[float]):
        self.connections = mp.solutions.holistic.HAND_CONNECTIONS

        landmarks = np.array(landmarks).reshape((21, 3))
        self.feature_vector = self._get_feature_vector(landmarks)

    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        """
        Params
            landmarks: numpy array of shape (21, 3)
        Return
            List of length nb_connections * nb_connections containing all the angles between the connections
        """
        connections = self._get_connections_from_landmarks(landmarks)

        angles_list = []
        for connection_from in connections:
            for connection_to in connections:

                angle = self._get_angle_between_vectors(connection_from, connection_to)
                # If the angle is not null we store it else we store 0
                if angle == angle:
                    angles_list.append(angle)
                else:
                    angles_list.append(0)
        return angles_list

    def _get_connections_from_landmarks(self, landmarks: np.ndarray) -> List[np.ndarray]:
        """
        Params
            landmarks: numpy array of shape (21, 3)
        Return
            List of vectors representing hand connections
        """
        return list(
            map(
                lambda t: landmarks[t[1]] - landmarks[t[0]],
                self.connections,
            )
        )

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        """
        Params
            u, v: 3D vectors representing two connections
        Return
            Angle between the two vectors
        """
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        return np.arccos(dot_product / norm)