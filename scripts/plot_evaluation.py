import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_evaluation_results(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        input_path = input_path / "evaluation.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        raw_results = json.load(handle)

    df = pd.json_normalize(raw_results)
    df = df.rename(columns=lambda name: name.replace("metrics.", ""))

    return df


def plot_metric_bars(df: pd.DataFrame, x_col: str, metrics: list[str], title: str, output_path: Path) -> None:
    x_labels = df[x_col].astype(str)
    figure, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics):
        x_positions = range(len(x_labels))
        ax.bar(x_positions, df[metric_name], color="#4c72b0")
        ax.set_title(metric_name.replace("@", " @ ").upper())
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_name)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        for index, value in enumerate(df[metric_name]):
            ax.text(index, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    figure.suptitle(title, fontsize=16)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def normalize_score(values: pd.Series, invert: bool = False) -> pd.Series:
    values = values.astype(float)
    if invert:
        values = 1.0 / values.replace(0, np.nan)

    minimum = values.min()
    maximum = values.max()
    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(0.0, index=values.index)
    if maximum == minimum:
        return pd.Series(0.5, index=values.index)

    return (values - minimum) / (maximum - minimum)


def plot_tradeoff_radar(
    df: pd.DataFrame,
    model_col: str,
    metrics: list[str],
    metric_labels: list[str],
    invert_metrics: list[bool],
    title: str,
    output_path: Path,
) -> None:
    radar_df = df.copy()
    for metric, invert in zip(metrics, invert_metrics):
        radar_df[metric] = normalize_score(radar_df[metric], invert=invert)

    categories = metric_labels
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    figure = plt.figure(figsize=(9, 9))
    ax = figure.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)

    for _, row in radar_df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row[model_col], linewidth=2)
        ax.fill(angles, values, alpha=0.15)

    ax.set_title(title, y=1.12, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model evaluation results from evaluation.json.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("experiments/results/exp_001/exp_001_20260402_195550/evaluation.json"),
        help="Path to the evaluation.json file or experiment directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".").resolve(),
        help="Directory where plots should be saved.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_evaluation_results(args.input)
    if "model" not in df.columns:
        raise ValueError("Expected 'model' column in evaluation results.")

    expected_metrics = ["ndcg@5", "precision@5", "latency_ms", "model_size_mb"]
    for metric_name in expected_metrics:
        if metric_name not in df.columns:
            raise ValueError(f"Missing expected metric column {metric_name} in evaluation results.")

    df = df.sort_values(by="ndcg@5", ascending=False).reset_index(drop=True)

    metric_plot_path = output_dir / "evaluation_metrics.png"
    plot_metric_bars(df, x_col="model", metrics=["ndcg@5", "precision@5"], title="Model Evaluation Metrics", output_path=metric_plot_path)

    metadata_plot_path = output_dir / "evaluation_latency_size.png"
    plot_metric_bars(df, x_col="model", metrics=["latency_ms", "model_size_mb"], title="Model Latency and Size", output_path=metadata_plot_path)

    radar_plot_path = output_dir / "evaluation_tradeoff_radar.png"
    plot_tradeoff_radar(
        df,
        model_col="model",
        metrics=["ndcg@5", "precision@5", "latency_ms", "model_size_mb"],
        metric_labels=["NDCG@5", "Precision@5", "Latency Efficiency", "Size Efficiency"],
        invert_metrics=[False, False, True, True],
        title="Predictive Power vs Resource Efficiency",
        output_path=radar_plot_path,
    )

    print(
        f"Saved evaluation plots:\n - {metric_plot_path}\n - {metadata_plot_path}\n - {radar_plot_path}"
    )


if __name__ == "__main__":
    main()
