import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VEBF import IrisDatasetLoader, Classification2DPlotter, run_experiment

def main():
    # Use Iris loader as default
    loader = IrisDatasetLoader(feature_indices=[2, 3])

    # Use Classification2DPlotter
    plotter = Classification2DPlotter()

    # Run experiment with standard VEBF
    print("\n--- Running Standard VEBF ---")
    run_experiment(loader, plotter, network_type="vebf")

    # Run experiment with Hybrid VEBF
    print("\n--- Running Hybrid VEBF ---")
    # Reset loader or ensure reusable (loader.load() usually shuffles, so might get slightly different splits if randomized.
    # To compare strictly, we might want fixed split, but for demo separate runs are fine)
    run_experiment(loader, plotter, network_type="hybrid")


if __name__ == "__main__":
    main()
