import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VEBF import PCAPlotter, BreastCancerDatasetLoader, run_experiment

def main():
    # Use BreastCancer loader
    loader = BreastCancerDatasetLoader(test_size=0.3, normalize=True)

    # Use PCAPlotter
    plotter = PCAPlotter(n_components=2)

    # Custom params for high dim
    params = {"delta": 2.0, "n0": 5, "theta": 0.9}

    run_experiment(loader, plotter, network_params=params)

if __name__ == "__main__":
    main()
