import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir   = "Predictive_model_data_archive/Bayes_results_XGB_HyperP_using_dP_for_delta_L_D"
output_dir = "Predictive_model_data/old_bayes_results"
cutoff     = 100


def plot_bayes_convergence_overlay(data_dir, output_dir, cutoff=100):
    csvs = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    rng  = np.random.default_rng(seed=42)
    line_colors = rng.random((len(csvs), 3)) * 0.7  # keep values dark enough to see

    fig, ax = plt.subplots(figsize=(6.5, 8))

    gains = []
    for i, fname in enumerate(csvs):
        df   = pd.read_csv(os.path.join(data_dir, fname))
        rmse = -df["target"].values
        running_best = np.minimum.accumulate(rmse)

        trials = np.arange(1, len(rmse) + 1)
        ax.plot(trials, running_best, color=line_colors[i],
                linewidth=1.0, alpha=0.75)

        gains.append(running_best[cutoff - 1] - running_best[-1])

    # cutoff marker
    ax.axvline(x=cutoff, color="black", linestyle="--", linewidth=1.2,
               label=f"Trial {cutoff} cutoff")
    ax.axvspan(cutoff, len(trials), color="grey", alpha=0.07)

    # annotation showing mean, median, max gain after cutoff
    median_gain = np.median(gains)
    mean_gain   = np.mean(gains)
    max_gain    = np.max(gains)
    ax.text(cutoff + 15, ax.get_ylim()[1] * 0.6,
            f"$\\Delta$RMSE after trial {cutoff}:\n"
            f"  Median: {median_gain:.4f}\n"
            f"  Mean:   {mean_gain:.4f}\n"
            f"  Max:    {max_gain:.4f}",
            fontsize=7, va="center", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.6))

    ax.set_xlabel("Trial number", fontsize=8)
    ax.set_ylabel("Running-best RMSE", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Bayes_convergence_all_predictors_absolute.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved: {out_path}")
    print(f"RMSE gain after trial {cutoff} — "
          f"median: {median_gain:.4f}, mean: {mean_gain:.4f}, "
          f"max: {max_gain:.4f}, min: {min(gains):.4f}")


if __name__ == "__main__":
    plot_bayes_convergence_overlay(data_dir, output_dir, cutoff=cutoff)
