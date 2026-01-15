import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

class Plotter(ABC):
    @abstractmethod
    def plot(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, network, feature_names: Optional[List[str]] = None, title: str = "VEBF Network Results"):
        pass
