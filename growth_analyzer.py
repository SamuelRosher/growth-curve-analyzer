import argparse
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit

from models import gompertz

COLORS = [
    '#0072B2',
    '#D55E00',
    '#009E73',
    '#CC79A7',
    '#E69F00',
    '#56B4E9',
    '#F0E442',
]


def load_data(filepath, time_col="time_h"):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        sys.exit(f"Error: could not find file '{filepath}'")

    if time_col not in df.columns:
        sys.exit(
            f"Error: expected a column named '{time_col}' but found: {list(df.columns)}\n"
            f"Use --time-col to specify a different column name."
        )

    time = df[time_col].values
    od_cols = [c for c in df.columns if c != time_col]
    return time, df[od_cols]


def group_replicates(data_df):
    replicate_pattern = re.compile(r'^(.+)_(\d+)$')
    groups = {}
    ungrouped = []

    for col in data_df.columns:
        match = replicate_pattern.match(col)
        if match:
            base_name = match.group(1)
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(col)
        else:
            ungrouped.append(col)

    result = {}
    for base_name, cols in groups.items():
        if len(cols) > 1:
            result[base_name] = (
                data_df[cols].mean(axis=1).values,
                data_df[cols].std(axis=1).values,
                len(cols)
            )
        else:
            result[base_name] = (data_df[cols[0]].values, None, 1)

    for col in ungrouped:
        result[col] = (data_df[col].values, None, 1)

    return result


def fit_gompertz(time, od_values):
    K_guess   = od_values.max()
    mu_guess  = 0.5
    lam_guess = time[len(time) // 5]

    p0     = [K_guess, mu_guess, lam_guess]
    bounds = (
        [0,           0,   0         ],
        [K_guess * 2, 10,  time.max()]
    )

    try:
        popt, pcov = curve_fit(
            gompertz, time, od_values,
            p0=p0, bounds=bounds, maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, pcov
    except RuntimeError:
        print("  Warning: curve fit did not converge.")
        return None, None, None


def calculate_r_squared(time, od_values, popt):
    od_predicted  = gompertz(time, *popt)
    ss_residuals  = np.sum((od_values - od_predicted) ** 2)
    ss_total      = np.sum((od_values - np.mean(od_values)) ** 2)
    return 1 - (ss_residuals / ss_total)


def calculate_params(popt, perr):
    K, mu_max, lam = popt
    K_err, mu_err, lam_err = perr

    td     = np.log(2) / mu_max
    td_err = (np.log(2) / mu_max ** 2) * mu_err

    return {
        "K (max OD)":           round(K,       4),
        "K stderr":             round(K_err,   4),
        "mu_max (per hr)":      round(mu_max,  4),
        "mu_max stderr":        round(mu_err,  4),
        "lag phase (hr)":       round(lam,     4),
        "lag stderr":           round(lam_err, 4),
        "doubling time (hr)":   round(td,      4),
        "doubling time stderr": round(td_err,  4),
    }


def plot_results(time, grouped, fit_results, output_path=None):
    plt.rcParams.update({
        'font.family':    'sans-serif',
        'font.size':      11,
        'axes.labelsize': 12,
        'axes.titlesize': 10,
        'xtick.labelsize':10,
        'ytick.labelsize':10,
        'legend.fontsize': 9,
        'legend.frameon': False,
    })

    n = len(grouped)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), sharey=True)

    if n == 1:
        axes = [axes]

    t_smooth     = np.linspace(time.min(), time.max(), 300)
    panel_labels = 'ABCDEFGHIJ'

    for idx, (ax, (condition, (mean_od, std_od, n_reps)), color, result) in \
            enumerate(zip(axes, grouped.items(), COLORS, fit_results)):

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Shaded SD region
        if std_od is not None:
            ax.fill_between(
                time,
                mean_od - std_od,
                mean_od + std_od,
                alpha=0.05, color=color
            )

        # Error bars on each data point
        ax.errorbar(
            time, mean_od,
            yerr=std_od if std_od is not None else None,
            fmt='o', color=color, markersize=3, zorder=4,
            capsize=3, capthick=1, elinewidth=1,
            ecolor='black', alpha=0.9
        )

        # Fitted curve + 95% CI
        if result["popt"] is not None and result["pcov"] is not None:
            K, mu_max, lam = result["popt"]

            od_fit = gompertz(t_smooth, K, mu_max, lam)
            ax.plot(t_smooth, od_fit, color=color, linewidth=1.2, zorder=3)

            try:
                samples = np.random.multivariate_normal(
                    result["popt"], result["pcov"], 1000
                )
                valid_curves = [
                    gompertz(t_smooth, *s)
                    for s in samples if all(s > 0)
                ]
                if valid_curves:
                    ci_low  = np.percentile(valid_curves, 2.5,  axis=0)
                    ci_high = np.percentile(valid_curves, 97.5, axis=0)
                    ax.fill_between(
                        t_smooth, ci_low, ci_high,
                        alpha=0.15, color=color
                    )
            except Exception:
                pass

            ax.axvline(lam, color='#666666', linestyle='--',
                       linewidth=1, alpha=0.8, zorder=2)
            ax.text(
                lam + 0.3, ax.get_ylim()[0] + 0.01,
                f'λ = {lam:.1f} h',
                color='#555555', fontsize=8, style='italic'
            )

            r2 = calculate_r_squared(time, mean_od, result["popt"])
            ax.text(
                0.97, 0.05, f'R² = {r2:.4f}',
                transform=ax.transAxes,
                fontsize=8, ha='right', color='#444444'
            )

        ax.text(
            -0.10, 1.08, panel_labels[idx],
            transform=ax.transAxes,
            fontsize=14, fontweight='bold'
        )

        p = result["params"]
        if p:
            subtitle = (
                f"K = {p['K (max OD)']} OD  |  "
                f"μmax = {p['mu_max (per hr)']} h⁻¹  |  "
                f"td = {p['doubling time (hr)']} h"
            )
        else:
            subtitle = "fit failed"

        display_name = condition.replace('_', ' ')
        ax.set_title(f"{display_name}\n{subtitle}", fontsize=9, pad=6)
        ax.set_xlabel('Time (h)', fontweight='bold')

        if idx == 0:
            ax.set_ylabel('OD₆₀₀', fontweight='bold')

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    ax0 = axes[0]
    legend_elements = [
        mpatches.Patch(facecolor=COLORS[0], alpha=0.20, label='± SD'),
        plt.Line2D([0], [0], color=COLORS[0], linewidth=2, label='Gompertz fit'),
        mpatches.Patch(facecolor=COLORS[0], alpha=0.15, label='95% CI (fit)'),
    ]
    ax0.legend(handles=legend_elements, loc='upper left')

    fig.suptitle('Bacterial Growth Curve Analysis',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def analyze_growth(filepath, time_col="time_h", plot=True,
                   save_plot=None, save_csv=None):

    print(f"\nLoading: {filepath}")
    time, data_df = load_data(filepath, time_col)

    grouped = group_replicates(data_df)
    print(f"Conditions found: {list(grouped.keys())}")
    print(f"Time range: {time.min():.1f} - {time.max():.1f} hr  "
          f"({len(time)} points)\n")

    fit_results  = []
    summary_rows = []

    for condition, (mean_od, std_od, n_reps) in grouped.items():
        n_str = f"{n_reps} replicate{'s' if n_reps > 1 else ''}"
        print(f"Fitting: {condition}  ({n_str})")

        popt, perr, pcov = fit_gompertz(time, mean_od)

        if popt is not None:
            params = calculate_params(popt, perr)
            r2     = calculate_r_squared(time, mean_od, popt)
            print(
                f"  K={params['K (max OD)']},  "
                f"mu_max={params['mu_max (per hr)']}/hr,  "
                f"lag={params['lag phase (hr)']} hr,  "
                f"t_d={params['doubling time (hr)']} hr,  "
                f"R²={round(r2, 4)}"
            )
            fit_results.append({"popt": popt, "params": params, "pcov": pcov})
            summary_rows.append({
                "condition":    condition,
                "n_replicates": n_reps,
                "R2":           round(r2, 4),
                **params
            })
        else:
            fit_results.append({"popt": None, "params": {}, "pcov": None})

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n── Summary ──────────────────────────────────────────────────────")
        print(
            summary_df[[
                "condition", "n_replicates", "R2", "K (max OD)",
                "mu_max (per hr)", "lag phase (hr)", "doubling time (hr)"
            ]].to_string(index=False)
        )

        if save_csv:
            summary_df.to_csv(save_csv, index=False)
            print(f"\nFull results saved to {save_csv}")

    if plot or save_plot:
        plot_results(time, grouped, fit_results, output_path=save_plot)

    return summary_df if summary_rows else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit Gompertz growth curves to bacterial OD600 data."
    )
    parser.add_argument("--file",      required=True)
    parser.add_argument("--time-col",  default="time_h")
    parser.add_argument("--plot",      action="store_true")
    parser.add_argument("--save-plot", metavar="PATH")
    parser.add_argument("--save-csv",  metavar="PATH")

    args = parser.parse_args()

    analyze_growth(
        filepath=args.file,
        time_col=args.time_col,
        plot=args.plot,
        save_plot=args.save_plot,
        save_csv=args.save_csv,
    )