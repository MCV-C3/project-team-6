import wandb
import pandas as pd
import numpy as np
from pathlib import Path

# ============ CONFIGURATION =============
WANDB_ENTITY = "mcv-team-6"
WANDB_PROJECT = "C3-Week2"

# Metric names in W&B
TEST_LOSS_KEY = "test_loss"
TEST_ACC_KEY = "test_accuracy"
TRAIN_LOSS_KEY = "train_loss"
TRAIN_ACC_KEY = "train_accuracy"

# Number of best epochs to average
TOP_N = 10

# List of txt files to analyze
TXT_FILES = [
    "report/first_experiments_with_augmentation.txt",
    "report/first_experiments_witouht_augmentation.txt",
    "report/depth_experiments_without_augmentation.txt",
    "report/depth_experiments_with_augmentation.txt",
    "report/experiment_best_depth_width.txt",
    "report/experiments_d5_image_size.txt",
    "report/experiments_w512_image_size.txt",
    "report/experiments_patch_based_patch_size.txt",
    "report/experiments_cool_cnns.txt",
]

# ========================================


def parse_experiment_file(file_path):
    """
    Parse a txt file to extract experiment names and run IDs.
    Returns a list of tuples: (experiment_name, run_id)
    """
    experiments = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by whitespace and take the last element as run_id
            parts = line.split()
            if len(parts) < 2:
                continue
            run_id = parts[-1]
            # Everything except the last part is the experiment name
            experiment_name = ' '.join(parts[:-1])
            experiments.append((experiment_name, run_id))
    return experiments


def get_run_metrics(api, run_id):
    """
    Fetch train and test metrics for a given run ID.
    Returns a tuple: (DataFrame with the metrics, list of tags)
    """
    try:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")

        # Get tags from the run
        tags = run.tags if hasattr(run, 'tags') else []

        # Scan history for both train and test metrics
        records = list(run.scan_history(keys=[TEST_ACC_KEY, TEST_LOSS_KEY, TRAIN_ACC_KEY, TRAIN_LOSS_KEY]))

        if len(records) == 0:
            print(f"⚠️  No history found for run {run_id}")
            return None, None

        df = pd.DataFrame(records)

        # Filter to only metrics we care about
        available_keys = [k for k in [TEST_ACC_KEY, TEST_LOSS_KEY, TRAIN_ACC_KEY, TRAIN_LOSS_KEY] if k in df.columns]

        if len(available_keys) == 0:
            print(f"⚠️  No metrics found for run {run_id}")
            return None, None

        return df[available_keys], tags

    except Exception as e:
        print(f"❌ Error fetching run {run_id}: {e}")
        return None, None


def calculate_top_n_average(df, metric_key, top_n, higher_is_better=True):
    """
    Calculate the average of the top N values for a given metric.

    Args:
        df: DataFrame with metric values
        metric_key: Name of the metric column
        top_n: Number of top values to average
        higher_is_better: If True, take the highest values; if False, take the lowest

    Returns:
        Average of top N values, or None if not enough data
    """
    if metric_key not in df.columns:
        return None

    values = df[metric_key].dropna()

    if len(values) == 0:
        return None

    # Take top N values
    n = min(top_n, len(values))
    if higher_is_better:
        top_values = values.nlargest(n)
    else:
        top_values = values.nsmallest(n)

    return top_values.mean()


def main():
    print(f"🚀 Analyzing experiments from W&B project: {WANDB_ENTITY}/{WANDB_PROJECT}")
    print(f"📊 Using top {TOP_N} epochs for averaging\n")

    api = wandb.Api()

    # Collect all experiments from all files
    all_experiments = []
    for txt_file in TXT_FILES:
        if not Path(txt_file).exists():
            print(f"⚠️  File not found: {txt_file}")
            continue

        print(f"📄 Reading {txt_file}...")
        experiments = parse_experiment_file(txt_file)
        all_experiments.extend(experiments)

    print(f"\n✅ Found {len(all_experiments)} experiments total\n")

    # Fetch metrics for each experiment
    results = []
    for exp_name, run_id in all_experiments:
        print(f"📥 Loading {exp_name} (run: {run_id})...")

        df, tags = get_run_metrics(api, run_id)

        if df is None:
            continue

        # Calculate top N averages for test metrics
        avg_test_acc = calculate_top_n_average(df, TEST_ACC_KEY, TOP_N, higher_is_better=True)
        avg_test_loss = calculate_top_n_average(df, TEST_LOSS_KEY, TOP_N, higher_is_better=False)

        # Calculate top N averages for train metrics
        avg_train_acc = calculate_top_n_average(df, TRAIN_ACC_KEY, TOP_N, higher_is_better=True)
        avg_train_loss = calculate_top_n_average(df, TRAIN_LOSS_KEY, TOP_N, higher_is_better=False)

        # Calculate best single epoch metrics
        best_test_acc = df[TEST_ACC_KEY].max() if TEST_ACC_KEY in df.columns else None
        best_test_loss = df[TEST_LOSS_KEY].min() if TEST_LOSS_KEY in df.columns else None

        # Calculate final epoch metrics (last valid value)
        final_test_acc = df[TEST_ACC_KEY].dropna().iloc[-1] if TEST_ACC_KEY in df.columns and len(df[TEST_ACC_KEY].dropna()) > 0 else None
        final_test_loss = df[TEST_LOSS_KEY].dropna().iloc[-1] if TEST_LOSS_KEY in df.columns and len(df[TEST_LOSS_KEY].dropna()) > 0 else None
        final_train_acc = df[TRAIN_ACC_KEY].dropna().iloc[-1] if TRAIN_ACC_KEY in df.columns and len(df[TRAIN_ACC_KEY].dropna()) > 0 else None
        final_train_loss = df[TRAIN_LOSS_KEY].dropna().iloc[-1] if TRAIN_LOSS_KEY in df.columns and len(df[TRAIN_LOSS_KEY].dropna()) > 0 else None

        # Calculate overfitting indicators (using top-N averages)
        acc_gap = None
        loss_gap = None
        if avg_train_acc is not None and avg_test_acc is not None:
            acc_gap = avg_train_acc - avg_test_acc  # Positive means overfitting
        if avg_train_loss is not None and avg_test_loss is not None:
            loss_gap = avg_test_loss - avg_train_loss  # Positive means overfitting

        results.append({
            'experiment_name': exp_name,
            'run_id': run_id,
            'tags': tags,
            'avg_top_test_accuracy': avg_test_acc,
            'avg_top_test_loss': avg_test_loss,
            'avg_top_train_accuracy': avg_train_acc,
            'avg_top_train_loss': avg_train_loss,
            'best_test_accuracy': best_test_acc,
            'best_test_loss': best_test_loss,
            'final_test_accuracy': final_test_acc,
            'final_test_loss': final_test_loss,
            'final_train_accuracy': final_train_acc,
            'final_train_loss': final_train_loss,
            'accuracy_gap': acc_gap,
            'loss_gap': loss_gap,
        })

        if avg_test_acc is not None:
            print(f"   ✓ Avg top-{TOP_N} test accuracy: {avg_test_acc:.4f}", end="")
            if avg_train_acc is not None:
                print(f" (train: {avg_train_acc:.4f}, gap: {acc_gap:.4f})")
            else:
                print()
        if avg_test_loss is not None:
            print(f"   ✓ Avg top-{TOP_N} test loss: {avg_test_loss:.4f}", end="")
            if avg_train_loss is not None:
                print(f" (train: {avg_train_loss:.4f}, gap: {loss_gap:.4f})")
            else:
                print()

    # Create DataFrame for analysis
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\n❌ No valid results found!")
        return
    
    N = 10
    
    print("\n" + "="*80)
    print(f"📈 TOP {N} MODELS BY TEST ACCURACY")
    print("="*80)

    # Rank by accuracy
    acc_ranked = results_df.dropna(subset=['avg_top_test_accuracy']).sort_values(
        'avg_top_test_accuracy', ascending=False
    )

    for i, row in enumerate(acc_ranked.head(N).itertuples(), 1):
        print(f"\n{i}. {row.experiment_name}")
        print(f"   Run ID: {row.run_id}")
        if row.tags:
            print(f"   Tags: {', '.join(row.tags)}")
        print(f"   📊 Test Metrics:")
        print(f"      Avg Top-{TOP_N} Accuracy: {row.avg_top_test_accuracy:.4f}")
        if pd.notna(row.avg_top_test_loss):
            print(f"      Avg Top-{TOP_N} Loss: {row.avg_top_test_loss:.4f}")
        if pd.notna(row.best_test_accuracy):
            print(f"      Best Accuracy: {row.best_test_accuracy:.4f}")
        if pd.notna(row.final_test_accuracy):
            print(f"      Final Accuracy: {row.final_test_accuracy:.4f}")

        print(f"   📊 Train Metrics:")
        if pd.notna(row.avg_top_train_accuracy):
            print(f"      Avg Top-{TOP_N} Accuracy: {row.avg_top_train_accuracy:.4f}")
        if pd.notna(row.avg_top_train_loss):
            print(f"      Avg Top-{TOP_N} Loss: {row.avg_top_train_loss:.4f}")
        if pd.notna(row.final_train_accuracy):
            print(f"      Final Accuracy: {row.final_train_accuracy:.4f}")
        if pd.notna(row.final_train_loss):
            print(f"      Final Loss: {row.final_train_loss:.4f}")

        print(f"   ⚠️  Overfitting Indicators:")
        if pd.notna(row.accuracy_gap):
            overfit_status = "⚠️ OVERFITTING" if row.accuracy_gap > 0.05 else "✓ Good"
            print(f"      Accuracy Gap (train-test): {row.accuracy_gap:+.4f} {overfit_status}")
        if pd.notna(row.loss_gap):
            overfit_status = "⚠️ OVERFITTING" if row.loss_gap > 0.1 else "✓ Good"
            print(f"      Loss Gap (test-train): {row.loss_gap:+.4f} {overfit_status}")

    print("\n" + "="*80)
    print(f"📉 TOP {N} MODELS BY TEST LOSS")
    print("="*80)

    # Rank by loss (lower is better)
    loss_ranked = results_df.dropna(subset=['avg_top_test_loss']).sort_values(
        'avg_top_test_loss', ascending=True
    )

    for i, row in enumerate(loss_ranked.head(N).itertuples(), 1):
        print(f"\n{i}. {row.experiment_name}")
        print(f"   Run ID: {row.run_id}")
        if row.tags:
            print(f"   Tags: {', '.join(row.tags)}")
        print(f"   📊 Test Metrics:")
        print(f"      Avg Top-{TOP_N} Loss: {row.avg_top_test_loss:.4f}")
        if pd.notna(row.avg_top_test_accuracy):
            print(f"      Avg Top-{TOP_N} Accuracy: {row.avg_top_test_accuracy:.4f}")
        if pd.notna(row.best_test_loss):
            print(f"      Best Loss: {row.best_test_loss:.4f}")
        if pd.notna(row.final_test_loss):
            print(f"      Final Loss: {row.final_test_loss:.4f}")

        print(f"   📊 Train Metrics:")
        if pd.notna(row.avg_top_train_accuracy):
            print(f"      Avg Top-{TOP_N} Accuracy: {row.avg_top_train_accuracy:.4f}")
        if pd.notna(row.avg_top_train_loss):
            print(f"      Avg Top-{TOP_N} Loss: {row.avg_top_train_loss:.4f}")
        if pd.notna(row.final_train_accuracy):
            print(f"      Final Accuracy: {row.final_train_accuracy:.4f}")
        if pd.notna(row.final_train_loss):
            print(f"      Final Loss: {row.final_train_loss:.4f}")

        print(f"   ⚠️  Overfitting Indicators:")
        if pd.notna(row.accuracy_gap):
            overfit_status = "⚠️ OVERFITTING" if row.accuracy_gap > 0.05 else "✓ Good"
            print(f"      Accuracy Gap (train-test): {row.accuracy_gap:+.4f} {overfit_status}")
        if pd.notna(row.loss_gap):
            overfit_status = "⚠️ OVERFITTING" if row.loss_gap > 0.1 else "✓ Good"
            print(f"      Loss Gap (test-train): {row.loss_gap:+.4f} {overfit_status}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
