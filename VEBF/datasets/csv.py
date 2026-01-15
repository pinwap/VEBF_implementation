import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from .base import DatasetLoader

class CSVDatasetLoader(DatasetLoader):
    def __init__(self, file_path, test_size=0.3, random_state=42, feature_names=None):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = feature_names

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        data = np.loadtxt(self.file_path, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]

        if self.feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
        else:
            feature_names = self.feature_names

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, feature_names
