import numpy as np
from sklearn.metrics import accuracy_score
from .vebf_network import VEBFNetwork
from .hybrid_vebf_network import HybridVEBFNetwork
from .datasets.base import DatasetLoader
from .plotting.base import Plotter
from typing import Optional, Dict, Any

def run_experiment(
    loader: DatasetLoader,
    plotter: Optional[Plotter] = None,
    network_params: Optional[Dict[str, Any]] = None,
    network_type: str = "vebf" # "vebf" or "hybrid"
):
    if network_params is None:
        network_params = {"delta": 1, "n0": 5, "theta": -0.5}

    print(f"Loading dataset using {loader.__class__.__name__}...")
    X_train, X_test, y_train, y_test, feature_names = loader.load()

    print(f"Features: {feature_names}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Create and train VEBF Network
    print(f"\n-----Starting training ({network_type})-----")
    print(f"Training on {X_train.shape[1]} dimensions...")
    
    if network_type.lower() == "hybrid":
        n_classes = len(np.unique(y_train))
        network = HybridVEBFNetwork(X_train=X_train, n_classes=n_classes, **network_params)
    else:
        network = VEBFNetwork(X_train=X_train, **network_params)

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
    if plotter:
         plotter.plot(X_train, y_train, X_test, y_test, network, feature_names=feature_names, title=f"{network_type.upper()} VEBF Result (Test Acc: {test_accuracy*100:.2f}%)")
    else:
        print("No plotter provided, skipping visualization.")

