import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import List, Optional
from .base import Plotter

class Classification2DPlotter(Plotter):
    def plot(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, network, feature_names: Optional[List[str]] = None, title: str = "VEBF Network Results"):
        # Combine train and test for visualization
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        if X.shape[1] != 2:
             print("Data is not 2D. Classification2DPlotter requires 2 features.")
             return

        plt.figure(figsize=(10, 6))

        # 1. Plot data points
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
        unique_classes = np.unique(y)
        for class_label in unique_classes:
            subset = X[y == class_label]
            color = colors[int(class_label)] if int(class_label) < len(colors) else 'gray'
            plt.scatter(subset[:, 0], subset[:, 1], c=color, label=f"Class {int(class_label)}", alpha=0.5, s=20)

        # 2. Plot neurons as ellipses
        ax = plt.gca()
        for neuron in network.neurons:
            center = neuron.center
            a = neuron.a
            eigenvectors = neuron.eigenvectors

            # Angle of the first eigenvector
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            # Create Ellipse patch
            color = colors[int(neuron.label)] if int(neuron.label) < len(colors) else 'black'
            ellipse = Ellipse(xy=center, width=2* a[0], height=2* a[1], angle=angle, edgecolor=color, facecolor='none', linewidth=2, linestyle='--')

            ax.add_patch(ellipse)
            # Center point
            plt.plot(center[0], center[1], 'x', color='black', markersize=8)

        plt.title(title)
        if feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()
