import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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

    # split train-test data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create and train VEBF Network
    print("\n-----Starting training-----")
    network = VebfNetwork(X_train=X_train, delta=1, n0=5, theta=-0.5)
    
    network.train(X_train, y_train)
    print("Training completed. Total neurons created:", len(network.neurons))
     
    # Evaluate accuracy
    print("\n-----Evaluation-----")
    train_pred = network.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    
    test_pred = network.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Plot results
    plot_result(X_selected, y, network, title=f"VEBF Result on Iris Dataset.(Test Acc: {test_accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()