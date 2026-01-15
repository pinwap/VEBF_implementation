import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Union
from .base import DatasetLoader

class IrisDatasetLoader(DatasetLoader):
    def __init__(self, test_size=0.3, random_state=42, feature_indices=None):
        self.test_size = test_size
        self.random_state = random_state
        self.feature_indices = feature_indices if feature_indices is not None else slice(None)

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        iris = datasets.load_iris()
        X = iris.data[:, self.feature_indices]
        y = iris.target

        # Get feature names based on indices
        all_feature_names = iris.feature_names
        if isinstance(self.feature_indices, list):
            feature_names = [all_feature_names[i] for i in self.feature_indices]
        elif isinstance(self.feature_indices, slice):
            # Convert slice to actual names
            all_indices = list(range(len(all_feature_names)))
            selected_indices = all_indices[self.feature_indices]
            feature_names = [all_feature_names[i] for i in selected_indices]
        else:
            feature_names = all_feature_names

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, feature_names
