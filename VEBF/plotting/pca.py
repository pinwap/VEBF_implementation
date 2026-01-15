import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Optional
from .base import Plotter

class PCAPlotter(Plotter):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def plot(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, network, feature_names: Optional[List[str]] = None, title: str = "VEBF Prediction (PCA Projection)"):
        # Use Test data for visualization validation
        X = X_test
        y = y_test

        # Predict labels for coloring (or use ground truth 'y')
        # In this context, 'y' passed is usually the ground truth.
        # But for high_dim_demo, we might want to visualize predictions vs truth.
        # Ideally, we should receive predictions or the network to predict.

        y_pred = network.predict(X)
        acc = np.mean(y == y_pred)

        print(f"\nVisualizing with PCA ({self.n_components}D Projection)...")
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6, edgecolors='k')

        # Plot errors
        incorrect_idx = np.where(y_pred != y)[0]
        if len(incorrect_idx) > 0:
            plt.scatter(X_pca[incorrect_idx, 0], X_pca[incorrect_idx, 1],
                        marker='x', c='black', s=100, linewidth=2, label='Prediction Error')

        plt.title(f'{title}\nAccuracy: {acc*100:.2f}%')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.grid(True, alpha=0.3)
        plt.show()
