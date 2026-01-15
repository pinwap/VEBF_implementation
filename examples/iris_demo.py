import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VEBF import IrisDatasetLoader, Classification2DPlotter, run_experiment

def main():
    # Use Iris loader as default
    loader = IrisDatasetLoader(feature_indices=[2, 3])

    # Use Classification2DPlotter
    plotter = Classification2DPlotter()

    # Run experiment
    run_experiment(loader, plotter)

if __name__ == "__main__":
    main()
