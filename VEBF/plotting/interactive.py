import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from typing import List, Optional
from .base import Plotter

class InteractivePlotter(Plotter):
    def __init__(self, use_pca: bool = False, n_pca_components: int = 3):
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components

    def plot(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, network, feature_names: Optional[List[str]] = None, title: str = "VEBF Interactive Result"):
        print("Generating interactive plot...")

        # Combine data for visualization
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        indices = np.arange(len(y))

        # Create metadata for hover
        split_labels = ["Train"] * len(y_train) + ["Test"] * len(y_test)

        # Determine plotting strategy
        if self.use_pca or X.shape[1] > 3:
            title += " (PCA Projection)"
            pca = PCA(n_components=self.n_pca_components)
            X_plot = pca.fit_transform(X)

            common_args = dict(
                color=y.astype(str),
                symbol=split_labels,
                title=title,
                labels={'color': 'Class', 'symbol': 'Split'},
                opacity=0.7,
                hover_data={"Index": indices}
            )

            if self.n_pca_components == 3:
                fig = px.scatter_3d(
                    x=X_plot[:, 0], y=X_plot[:, 1], z=X_plot[:, 2],
                    **common_args
                )
            else:
                 fig = px.scatter(
                    x=X_plot[:, 0], y=X_plot[:, 1],
                    **common_args
                )
        else:
            X_plot = X # For consistent variable name below
            # Native Dimensionality (2D or 3D)
            common_args = dict(
                color=y.astype(str),
                symbol=split_labels,
                title=title,
                labels={'color': 'Class', 'symbol': 'Split'},
                hover_data={"Index": indices}
            )

            if X.shape[1] == 2:
                labels = {
                    "x": feature_names[0] if feature_names else "Feature 1",
                    "y": feature_names[1] if feature_names else "Feature 2",
                    "color": "Class",
                    "symbol": "Split"
                }
                common_args['labels'] = labels

                fig = px.scatter(
                    x=X[:, 0], y=X[:, 1],
                    **common_args
                )

                # Add neurons as shapes (approximations)
                neuron_centers = np.array([n.center for n in network.neurons])
                neuron_labels = np.array([n.label for n in network.neurons])

                fig.add_trace(go.Scatter(
                    x=neuron_centers[:, 0],
                    y=neuron_centers[:, 1],
                    mode='markers',
                    marker=dict(size=12, symbol='x-thin', line=dict(width=2, color='black')),
                    name='Neuron Centers',
                    text=[f"Neuron Class: {l}" for l in neuron_labels]
                ))

            elif X.shape[1] == 3:
                 fig = px.scatter_3d(
                    x=X[:, 0], y=X[:, 1], z=X[:, 2],
                    **common_args
                )
            else:
                print("Data > 3D and PCA disabled. Cannot plot.")
                return

        # Add Predictions Errors (Linked to Traces)
        y_pred = network.predict(X)
        incorrect_mask = y != y_pred

        # We need to iterate over the *existing* traces to create matching error traces
        # This ensures that when a user hides "Class 0", the errors for Class 0 also hide.
        # PX stores the source DataFrame index in customdata if we pass hover_data

        new_traces = []
        for trace in fig.data:
            if not hasattr(trace, 'customdata') or trace.customdata is None:
                continue

            # customdata is usually [[idx], [idx], ...] from px
            # Extract indices for this trace
            try:
                # Handle different shapes/types of customdata from various PX versions
                trace_indices = np.array([c[0] for c in trace.customdata])
            except (IndexError, TypeError):
                continue

            # Find which of these are errors
            # We must be careful with indexing: trace_indices are indices into our original X array
            trace_errors_mask = incorrect_mask[trace_indices]

            if np.any(trace_errors_mask):
                error_indices = trace_indices[trace_errors_mask]

                # Get coordinates for these errors
                # Use X_plot which is already projected if needed
                X_error = X_plot[error_indices]

                # Construct Error Trace
                error_trace_kwargs = dict(
                    mode='markers',
                    marker=dict(size=2, symbol='x', color='black', line=dict(width=2)),
                    name=trace.name + " (Errors)",
                    legendgroup=trace.legendgroup, # LINKING HAPPENS HERE
                    showlegend=False, # Don't clutter legend, just follow parent
                    text=[f"True: {t}, Pred: {p}" for t, p in zip(y[error_indices], y_pred[error_indices])],
                    hovertemplate="<b>%{text}</b><extra></extra>"
                )

                if X_error.shape[1] == 3:
                     # For 3D, we need x, y, Z
                     new_traces.append(go.Scatter3d(x=X_error[:, 0], y=X_error[:, 1], z=X_error[:, 2], **error_trace_kwargs))
                else:
                    new_traces.append(go.Scatter(x=X_error[:, 0], y=X_error[:, 1], **error_trace_kwargs))

        for t in new_traces:
            fig.add_trace(t)

        fig.show()
