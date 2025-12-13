import wandb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# ============ CONFIGURATION =============
# Change these if your metric names differ
TRAIN_LOSS_KEY  = "train_loss"
TEST_LOSS_KEY   = "test_loss"
TRAIN_ACC_KEY   = "train_accuracy"
TEST_ACC_KEY    = "test_accuracy"
X_KEY           = "epoch"   # or "step" if you logged that

# Smoothing factor (0 = no smoothing, 1 = heavy)
SMOOTH_ALPHA = 0.1

# ========================================


def smooth(series, x_values=None, alpha=0.6):
    if len(series) == 0:
        return []

    out = []
    last = series[0]

    if x_values is not None and len(x_values) == len(series):
        for i, x in enumerate(series):
            if i == 0:
                out.append(x)
            else:
                delta = x_values[i] - x_values[i-1]
                adjusted_alpha = 1 - (1 - alpha) ** delta
                last = adjusted_alpha * x + (1 - adjusted_alpha) * last
                out.append(last)
    else:
        for x in series:
            last = alpha * x + (1 - alpha) * last
            out.append(last)

    return out


def load_history_for_run(run, keys):
    """
    Load history for a given run using scan_history
    (which fetches *all* metrics, not a sampled version).
    """
    # API run object
    # First try with X_KEY (epoch), if that fails, try without it
    records = list(run.scan_history(keys=keys + [X_KEY]))

    if len(records) == 0:
        # X_KEY might not exist, try without it
        records = list(run.scan_history(keys=keys))

    if len(records) == 0:
        # Still no records? Try getting all history without specifying keys
        records = list(run.scan_history())

    if len(records) == 0:
        print(f"⚠️ No history found for run {run.id}")
        return None

    df = pd.DataFrame(records)

    # Filter to only the keys we care about (plus X_KEY if it exists)
    available_keys = [k for k in keys if k in df.columns]
    if X_KEY in df.columns:
        available_keys.append(X_KEY)

    if len(available_keys) == 0:
        print(f"⚠️ No requested metrics found in run {run.id}")
        return None

    df = df[available_keys]

    if X_KEY not in df:
        # if the epoch key was not logged for each step,
        # just use index as x
        df[X_KEY] = df.index

    return df.sort_values(by=X_KEY)


def plot_metric(all_runs_data, metric_train, metric_test, save_as, max_epochs=None):
    """
    Plot train and test curves for the given metric across runs

    Args:
        all_runs_data: Dictionary of run_name -> DataFrame
        metric_train: Name of the training metric
        metric_test: Name of the test metric
        save_as: Filename to save the plot
        max_epochs: Maximum number of epochs to plot (None = plot all)
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")

    best_val = None
    best_run = None

    # Get a color palette for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_runs_data)))

    for idx, (run_name, df) in enumerate(all_runs_data.items()):
        # Only plot if metrics exist
        if metric_train not in df or metric_test not in df:
            continue

        # Filter by max_epochs if specified
        if max_epochs is not None:
            df = df[df[X_KEY] <= max_epochs]

        x = df[X_KEY].values

        y_train = df[metric_train].values
        y_test  = df[metric_test].values

        # Time-weighted smoothing
        y_train_sm = smooth(y_train, x_values=x, alpha=SMOOTH_ALPHA)
        y_test_sm  = smooth(y_test, x_values=x, alpha=SMOOTH_ALPHA)

        # Use the same color for train and test of the same run
        color = colors[idx]

        # Plot original lightly
        # plt.plot(x, y_train, color=color, alpha=0.15, linestyle="--", linewidth=1)
        # plt.plot(x, y_test,  color=color, alpha=0.15, linestyle="-", linewidth=1)

        # Plot smoothed - train is dashed, test is solid
        plt.plot(x, y_train_sm, color=color, label=run_name, linestyle="--", linewidth=2, alpha=0.7)
        plt.plot(x, y_test_sm,  color=color, linestyle="-", linewidth=2)

        # Track best values (for metric choice)
        if "loss" in metric_train.lower():
            # lower loss == better
            min_test = np.min(y_test)
            if best_val is None or min_test < best_val:
                best_val = min_test
                best_run = run_name
        else:
            # higher accuracy == better
            max_test = np.max(y_test)
            if best_val is None or max_test > best_val:
                best_val = max_test
                best_run = run_name

    plt.title(f"{metric_train} & {metric_test}")
    plt.xlabel("Epoch")

    # Add custom legend explanation
    legend = plt.legend(title="Run (--train / —test)")
    plt.setp(legend.get_title(), fontsize=9)

    plt.tight_layout()

    if best_run is not None:
        plt.annotate(
            f"Best: {best_run}",
            xy=(0.95, 0.01), xycoords="axes fraction",
            fontsize=12, ha="right"
        )

    plt.savefig(save_as)
    plt.show()


def main(entity_project, run_ids=None, max_epochs=None):
    """
    entity_project: "username/projectname"
    run_ids: optional list of specific run IDs to plot
    max_epochs: optional maximum number of epochs to plot
    """
    api = wandb.Api()

    # fetch runs
    runs = api.runs(entity_project)
    if run_ids:
        runs = [r for r in runs if r.id in run_ids]

    # Load histories
    all_runs_data = {}
    for run in runs:
        print(f"📥 Loading run {run.id} ({run.name}) ...")
        df = load_history_for_run(run, [
            TRAIN_LOSS_KEY, TEST_LOSS_KEY,
            TRAIN_ACC_KEY, TEST_ACC_KEY
        ])
        if df is not None:
            all_runs_data[run.name or run.id] = df

    # Plot metrics
    plot_metric(all_runs_data, TRAIN_ACC_KEY, TEST_ACC_KEY, "accuracy_plot.png", max_epochs=max_epochs)
    plot_metric(all_runs_data, TRAIN_LOSS_KEY, TEST_LOSS_KEY, "loss_plot.png", max_epochs=max_epochs)

    print("✅ Plots saved as accuracy_plot.png and loss_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        required=True,
        help="W&B project in form 'entity/project'"
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Specific run IDs to include (optional)."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum number of epochs to plot (optional)."
    )
    args = parser.parse_args()

    main(args.project, args.runs, args.max_epochs)
