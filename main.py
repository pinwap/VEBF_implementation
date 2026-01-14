import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from vebf_network import VebfNetwork
from utils import calculate_initial_a, load_dataset

def plot_result(X, y, network, title="VEBF Network Results"):
    
    plt.figure(figsize=(10, 6))
    
    # 1. Plot data points
    colors = ['red', 'green', 'blue']
    for class_label in np.unique(y):
        subset = X[y == class_label]
        plt.scatter(subset[:, 0], subset[:, 1], c=colors[int(class_label)], label=f"Class {int(class_label)}", alpha=0.5, s=20)
        
    # 2. Plot neurons as ellipses
    ax = plt.gca()
    for neuron in network.neurons:
        center = neuron.center
        a = neuron.a
        eigenvectors = neuron.eigenvectors
        
        # หามุมของแกนหลักวงรี = Eigenvector ตัวแรก (eigenvectors[:, 0])
        # arctan2(y, x) คืนค่าเป็นเรเดียน
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # สร้าง Ellipse patch
        ellipse = Ellipse(xy=center, width=2* a[0], height=2* a[1], angle=angle, edgecolor=colors[int(neuron.label)], facecolor='none', linewidth=2, linestyle='--')
        
        ax.add_patch(ellipse)
        # จุดศูนย์กลาง
        plt.plot(center[0], center[1], 'x', color='black', markersize=8)
    
    plt.title(title)
    plt.xlabel("Feature 1 (Petal Length)")
    plt.ylabel("Feature 2 (Petal Width)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    
def main():
    # Load Iris dataset
    print("Loading Iris dataset...")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # เลือกแค่ 2 feature แรก = Petal Length, Petal Width จะได้วาดกราฟง่ายๆ
    X_selected = X[:, 2:4]
    
    # Shuffle data
    X_shuffled, y_shuffled = shuffle(X_selected, y, random_state=42)
    
    # Create and train VEBF Network
    print("-----Starting training-----")
    network = VebfNetwork(X_train=X_shuffled, delta=1.5, n0=5, theta=0.8)
    
    network.train(X_train=X_shuffled, y_train=y_shuffled)
    print("Training completed. Total neurons created:", len(network.neurons))
     
    # Evaluate accuracy
    y_pred = network.predict(X_shuffled)
    accuracy = accuracy_score(y_shuffled, y_pred)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    
    # Plot results
    plot_result(X_shuffled, y_shuffled, network, title="VEBF Network on Iris Dataset")

if __name__ == "__main__":
    main()