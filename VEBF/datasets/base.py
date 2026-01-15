import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List

class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load and return X_train, X_test, y_train, y_test, feature_names"""
        pass
