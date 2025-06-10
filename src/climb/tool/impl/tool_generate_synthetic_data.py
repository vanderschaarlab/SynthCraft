import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Literal
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pandas as pd

# synthcity
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader, DataLoader
from synthcity.benchmark import Benchmarks
from synthcity.utils.serialization import load_from_file
import synthcity.logger as log

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase

def find_synthetic_cache(
        workspace: str,
        syn_gen_method: str,
        cache_type: Literal["generator", "data", "augmentation", "augmentation_generator"] = "data"
) -> List[str]:
    """
    Recursively search the workspace directory for .bkp files whose
    name contains the method string, *excluding* any paths that
    contain "generator", "augmentation", or "reference".
    """
    workspace_path = Path(workspace)
    pattern = f"*{syn_gen_method}*.bkp"

    # find all matching cache files
    all_paths = [str(p) for p in workspace_path.rglob(pattern)]

    # filter out any paths with unwanted substrings
    if cache_type == "generator":
        cache_paths = [
            p for p in all_paths
            if "generator" in p.lower()
        ]
    elif cache_type == "data":
        exclude_substrings = ("generator", "augmentation", "reference")
        cache_paths = [
            p for p in all_paths
            if not any(sub in p.lower() for sub in exclude_substrings)
        ]
    elif cache_type == "augmentation":
        cache_paths = [
            p for p in all_paths
            if "augmentation" in p.lower() and "generator" not in p.lower()
        ]
    elif cache_type == "augmentation_generator":
        cache_paths = [
            p for p in all_paths
            if "augmentation" in p.lower() and "generator" in p.lower()
        ]

    if not cache_paths:
        raise FileNotFoundError(
            f"No synthetic data cache files found for method '{syn_gen_method}' "
            f"in workspace '{workspace}', after filtering out generator/augmentation/reference."
        )

    if len(cache_paths) > 1:
        print(f"Warning: Found multiple synthetic data cache files for method '{syn_gen_method}': {cache_paths}")
        print("Using the first one found.")

    return cache_paths[0]

def generate_synthetic_data(
    tc: ToolCommunicator,
    data_file_path: str,
    synthetic_data_file_path: str,
    target_column: str,
    syn_gen_methods: str,
    workspace: str,
    syn_count: Optional[int] = None,
    task_type: Optional[str] = "classification",
    augment_data: bool = False,
    fairness_column: Optional[str] = None,
) -> None:
    
    log.add(level="CRITICAL")  # Set logging level to critical to suppress most logs


    data_file_path = Path(workspace) / data_file_path

    df = pd.read_csv(data_file_path)
    loader = GenericDataLoader(
        df,
        target_column=target_column,
        fairness_column=fairness_column,
    )
    if syn_count is None:
        syn_count = len(df)

    syn_gen_methods = syn_gen_methods.lower().split(",")

    metrics = {
        'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
        'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
        'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
        'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']
    }
    if augment_data:
        tc.print(
            "Augmentation is enabled. Adding augmentation metrics to the evaluation."
            f"Performance metrics: {metrics['performance']} Fairness Column: {fairness_column}"
        )
        metrics['performance'] += ['xgb_augmentation', 'linear_model_augmentation', 'mlp_augmentation']

    score = Benchmarks.evaluate(
        [(f"{syn_gen_method}_copilot_evaluation", syn_gen_method, {}) for syn_gen_method in syn_gen_methods],
        loader,
        metrics=metrics,
        synthetic_size=syn_count,
        repeats=3,
        synthetic_cache= True,
        synthetic_reuse_if_exists=True,
        augmented_reuse_if_exists=True,
        task_type=task_type,
        workspace=workspace,
        augmentation_rule="equal",
        strict_augmentation=False,
        ad_hoc_augment_vals=None,
    )
    # Save the synthetic data to the specified file path
    syn_data_paths = []
    augmentation_paths = []
    for syn_gen_method in syn_gen_methods:
        X_syn_cache_file = find_synthetic_cache(workspace, syn_gen_method)
        # Synthetic data
        X_syn = load_from_file(X_syn_cache_file).dataframe()
        synthetic_data_file_path_specific = Path(synthetic_data_file_path).parent / f"{syn_gen_method}_{Path(synthetic_data_file_path).name}"
        X_syn.to_csv(synthetic_data_file_path_specific, index=False)
        syn_data_paths.append(str(synthetic_data_file_path_specific))
        # Augmentation data
        if augment_data:
            tc.print(
                "Augmentation is enabled. Saving augmentation data to csv files."
            )
            X_aug_cache_file = find_synthetic_cache(workspace, syn_gen_method, cache_type="augmentation")
            X_aug = load_from_file(X_aug_cache_file).dataframe()
            augmentation_file_path_specific = Path(synthetic_data_file_path).parent / f"{syn_gen_method}_augmentation_{Path(synthetic_data_file_path).name}"
            X_aug.to_csv(augmentation_file_path_specific, index=False)
            augmentation_paths.append(str(augmentation_file_path_specific))

    # Log the results
    tc.set_returns(
        tool_return=(
            f"The synthetic data has been generated and metrics evaluated for {', '.join(syn_gen_methods)}."
            f"Metrics: {Benchmarks.print(score)}."
            # f"Metrics: {Benchmarks.highlight(score)}."
            f"The Synthetic dataset has been saved to {', '.join(syn_data_paths)}."
            f"The Augmentation dataset has been saved to {', '.join(augmentation_paths)}." if augment_data else ""
        ),
        user_report=[
            f"The synthetic data has been generated and metrics evaluated for {', '.join(syn_gen_methods)}.",
            f"Metrics: {Benchmarks.print(score)}.",
            # f"Metrics: {Benchmarks.highlight(score)}.",
            f"The Synthetic dataset has been saved to {', '.join(syn_data_paths)}.",
        ],
    )

class SyntheticGenerator(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        out_path = os.path.join(self.working_directory, kwargs["synthetic_data_file_path"])
        thrd, out_stream = execute_tool(
            generate_synthetic_data,
            wd=self.working_directory,
            # ---
            data_file_path=real_path,
            synthetic_data_file_path=out_path,
            target_column=kwargs["target_column"],
            task_type=kwargs.get("task_type", "classification"),
            syn_gen_methods=kwargs["syn_gen_methods"],
            workspace=self.working_directory,
            syn_count=kwargs.get("syn_count", None),
            augment_data=kwargs.get("augment_data", False),
            fairness_column=kwargs.get("fairness_column", None),
        )
        self.tool_thread = thrd
        return out_stream

    # @property
    # def logs_useful(self) -> bool:
    #     return True

    @property
    def name(self) -> str:
        return "generate_synthetic_data"

    @property
    def description(self) -> str:
        return """
        Uses the `generate_synthetic_data` tool to generate synthetic data and benchmark it with in-built metrics.
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file_path": {"type": "string", "description": "Path to the data file."},
                        "synthetic_data_file_path": {
                            "type": "string",
                            "description": "Path to the data file with extracted features, which this function creates.",
                        },
                        "target_column": {
                            "type": "string",
                            "description": "The target column to predict in the research task. For survival analysis this should be the event column.",
                        },
                        "syn_gen_methods": {
                            "type": "string",
                            "description": "The synthetic data generation methods to use. It should be a comma delimited list of available plugins in synthcity.",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "The type of task for which the synthetic data is generated. It can be 'classification' or 'regression'.",
                            "enum": ["classification", "regression"],
                        },
                        "syn_count": {
                            "type": "integer",
                            "description": "The number of synthetic samples to generate. If not provided, it defaults to the number of samples in the original dataset.",
                        },
                        "augment_data": {
                            "type": "boolean",
                            "description": "Whether to augment the original data with synthetic data to improve fairness. Defaults to False.",
                        },
                        "fairness_column": {
                            "type": "string",
                            "description": "The fairness column to use for the research task. If not provided, it defaults to None. This MUST be provided if augment_data is True.",
                        },
                    },
                    "required": [
                        "data_file_path",
                        "synthetic_data_file_path",
                        "syn_gen_methods",
                        "target_column",
                        "task_type",
                        "augment_data",
                    ],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "Uses the `generate_synthetic_data` tool to generate synthetic data and benchmark it with in-built metrics."





def plot_tsne(
    plt: Any,
    X_gt: DataLoader,
    X_syn: DataLoader,
    syn_gen_method: str,
) -> Figure:
    """
    Plots t-SNE visualization of real and synthetic data and returns the Figure.

    Args:
        plt: Matplotlib pyplot module.
        X_gt: DataLoader for real data.
        X_syn: DataLoader for synthetic data.

    Returns:
        The matplotlib Figure containing the t-SNE scatter plot.
    """

    log.remove()

    # encode real & synthetic
    X_gt_enc, _ = X_gt.encode()
    X_syn_enc, _ = X_syn.encode()

    # create figure & axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # fit t-SNE on real
    tsne_gt = TSNE(n_components=2, random_state=0, learning_rate="auto", init="pca")
    proj_gt = pd.DataFrame(tsne_gt.fit_transform(X_gt_enc.dataframe()))

    # fit t-SNE on synthetic
    tsne_syn = TSNE(n_components=2, random_state=0, learning_rate="auto", init="pca")
    proj_syn = pd.DataFrame(tsne_syn.fit_transform(X_syn_enc.dataframe()))

    # plot both
    ax.scatter(x=proj_gt[0], y=proj_gt[1], s=10, label="Real data")
    ax.scatter(x=proj_syn[0], y=proj_syn[1], s=10, label="Synthetic data")

    ax.legend(loc="upper left")
    ax.set_ylabel("t-SNE component 2")
    ax.set_xlabel("t-SNE component 1")
    ax.set_title(f"{syn_gen_method} - t-SNE: Real vs Synthetic")

    # return the Figure so callers can include it in tc.set_returns(...)
    return fig



def plot_synthetic_data(
    tc: ToolCommunicator,
    data_file_path: str,
    target_column: str,
    syn_gen_methods: str,
    workspace: str,
    syn_count: Optional[int] = None,
) -> None:

    data_file_path = Path(workspace) / data_file_path

    df = pd.read_csv(data_file_path)

    loader = GenericDataLoader(
        df,
        target_column=target_column,
    )
    if syn_count is None:
        syn_count = len(df)

    syn_gen_methods = syn_gen_methods.lower().split(",")

    tsne_plots = []
    for syn_gen_method in syn_gen_methods:
        syn_model_cache_file = find_synthetic_cache(workspace, syn_gen_method, cache_type="data")
        X_syn = load_from_file(syn_model_cache_file)
        tsne_plots.append(
            plot_tsne(
                plt,
                X_gt=loader,
                X_syn=X_syn,
                syn_gen_method=syn_gen_method
            )
        )
    
    # Log the results
    tc.set_returns(
        tool_return=(
            f"The synthetic data has been plotted for {', '.join(syn_gen_methods)}. "
            f"The plots are available in the working directory: {workspace}."
        ),
        user_report=[
            f"The synthetic data has been plotted for {', '.join(syn_gen_methods)}.",
            "📈 **t-SNE comparison**",
        ] + [
            fig for fig in tsne_plots
        ],
    )


class PlotSyntheticData(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        thrd, out_stream = execute_tool(
            plot_synthetic_data,
            wd=self.working_directory,
            # ---
            data_file_path=real_path,
            target_column=kwargs["target_column"],
            syn_gen_methods=kwargs["syn_gen_methods"],
            workspace=self.working_directory,
            syn_count=kwargs.get("syn_count", None),
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "plot_synthetic_data"

    @property
    def description(self) -> str:
        return """
        Uses the `plot_synthetic_data` tool to plot synthetic data generated by the `generate_synthetic_data` tool.
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file_path": {"type": "string", "description": "Path to the data file."},
                        "target_column": {
                            "type": "string",
                            "description": "The target column to predict in the research task. For survival analysis this should be the event column.",
                        },
                        "syn_gen_methods": {
                            "type": "string",
                            "description": "The synthetic data generation methods to use. It should be a comma delimited list of available plugins in synthcity.",
                        },
                        "syn_count": {
                            "type": "integer",
                            "description": "The number of synthetic samples to generate. If not provided, it defaults to the number of samples in the original dataset.",
                        },
                    },
                    "required": [
                        "data_file_path",
                        "syn_gen_methods",
                        "target_column",
                    ],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "Uses the `plot_synthetic_data` tool to plot synthetic data generated by the `generate_synthetic_data` tool."

