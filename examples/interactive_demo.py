import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VEBF import BreastCancerDatasetLoader, InteractivePlotter, run_experiment

def main():
    print("Running Interactive VEBF Demo...")

    # Use BreastCancer loader
    loader = BreastCancerDatasetLoader(test_size=0.3, normalize=True)

    # Use NEW InteractivePlotter (with PCA enabled for high dim)
    # This will open a browser window with a 3D scatter plot
    plotter = InteractivePlotter(use_pca=True, n_pca_components=3)

    # Custom params
    params = {"delta": 2.0, "n0": 5, "theta": 0.9}

    run_experiment(loader, plotter, network_params=params)

if __name__ == "__main__":
    main()
