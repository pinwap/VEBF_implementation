import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from .base import DatasetLoader

class BreastCancerDatasetLoader(DatasetLoader):
    def __init__(self, test_size=0.3, random_state=42, normalize=True):
        self.test_size = test_size
        self.random_state = random_state
        self.normalize = normalize

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names

        if self.normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, list(feature_names)
