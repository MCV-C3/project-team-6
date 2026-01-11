import wandb
import pandas as pd

# Configuration
WANDB_ENTITY = "mcv-team-6"
WANDB_PROJECT = "C3-Week2"

# Run IDs from experiments_pyramidal.txt
run_ids = {
    "Pyramidal gran_denormal": "f4axygz6",
    "Pyramidal petit_normal": "es0u0hlp",
    "Pyramidal petit_denormal": "xqvn6f96",
    "Pyramidal gran_normal": "axlvhge7"
}

def export_pyramidal_data():
    api = wandb.Api()

    all_data = {}

    for exp_name, run_id in run_ids.items():
        print(f"Fetching {exp_name} (run: {run_id})...")

        try:
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")

            # Get history with test and train metrics
            history = run.scan_history(keys=["test_accuracy", "train_accuracy", "test_loss", "train_loss", "_step"])

            records = list(history)
            if len(records) == 0:
                print(f"  ⚠️  No history found for {run_id}")
                continue

            df = pd.DataFrame(records)

            # Rename columns to include experiment name
            for col in df.columns:
                if col != "_step":
                    all_data[f"{exp_name} - {col}"] = df[col]

            # Use _step as the common index (Step column)
            if "Step" not in all_data and "_step" in df.columns:
                all_data["Step"] = df["_step"]

            print(f"  ✓ Fetched {len(df)} epochs")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    # Create combined dataframe
    df_combined = pd.DataFrame(all_data)

    # Save to CSV
    output_file = "wandb_export_pyramidal.csv"
    df_combined.to_csv(output_file, index=False)
    print(f"\n✅ Exported to {output_file}")
    print(f"   Shape: {df_combined.shape}")
    print(f"   Columns: {list(df_combined.columns)}")

    return df_combined

if __name__ == "__main__":
    export_pyramidal_data()
