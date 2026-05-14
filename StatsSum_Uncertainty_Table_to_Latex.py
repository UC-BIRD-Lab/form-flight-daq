import os
import pandas as pd
import numpy as np

l_d_stats_paths = [
    "Post_processed_stats_summarized/L_D_True_Stats_Iter_0.csv",
    "Post_processed_stats_summarized/L_D_True_Stats_Iter_1.csv",
]
pressure_stats_paths = [
    "Post_processed_stats_summarized/Pressure_True_Stats_Iter_0.csv",
    "Post_processed_stats_summarized/Pressure_True_Stats_Iter_1.csv",
]

os.makedirs("Post_processed_stats_summarized/Uncertainty", exist_ok=True)

output_tex = "Post_processed_stats_summarized/Uncertainty/Uncertainty_Table.tex"
force_coeff_path = "Post_processed_force_coefficient/Force_Coefficients_AoA_All.csv"
baseline_pressure_self_path = "Post_processed_stats_summarized/Baseline_Pressure_Self_Stats.csv"

force_units = "N"
pressure_units = "Pa"

upper_surface_taps = [0, 2, 4, 6, 8, 9, 10, 11, 13, 15, 17, 19]


def load_concat(paths):
    dfs = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def col_stats(df, col):
    v = df[col].dropna()
    return v.mean(), v.min(), v.max()


def fmt(v):
    return f"{v:.4f}"


def make_row(category, quantity, source, mean_val, min_val, max_val, units):
    return [category, quantity, source,
            fmt(mean_val), f"{fmt(min_val)} -- {fmt(max_val)}", units]


def make_na_row(category, quantity):
    return [category, quantity, "---", "---", "---", "---"]


def make_spec_row(category, quantity, source, spec_val, units):
    return [category, quantity, source, fmt(spec_val), "---", units]


def build_table(df_ld, df_p, df_fc=None, df_bp=None):
    rows = []

    # ------------------------------------------------------------------
    # Equipment specifications (manufacturer)
    # ------------------------------------------------------------------
    rows.append(make_spec_row(
        "Equipment", r"$F_{x,y}$ (load cell)",
        "ATI Mini40; manufacturer precision",
        1 / 50, force_units
    ))
    rows.append(make_spec_row(
        "Equipment", r"$F_z$ (load cell)",
        "ATI Mini40; manufacturer precision",
        1 / 25, force_units
    ))
    rows.append(make_spec_row(
        "Equipment", "Torque (load cell)",
        "ATI Mini40; manufacturer precision",
        1 / 2000, "Nm"
    ))
    rows.append(make_spec_row(
        "Equipment", "Pitot transducer",
        r"Omega PX2650; $\pm$1\% of $\pm$1244.2\,Pa range",
        0.01 * 1244.2, pressure_units
    ))
    rows.append(make_spec_row(
        "Equipment", "Tap transducer",
        r"Omega PX165-001B; $\pm$1\% of $\pm$248.8\,Pa range",
        0.01 * 248.8, pressure_units
    ))

    # ------------------------------------------------------------------
    # A: Physical setup / positioning
    # ------------------------------------------------------------------
    ruler_resolution_in = 1 / 16
    position_uncertainty = ruler_resolution_in / 2
    rows.append(make_row(
        "Positioning", r"$\Delta x,\, \Delta y,\, \Delta z$",
        "Half of ruler smallest division (1/16 in.)",
        position_uncertainty, position_uncertainty, position_uncertainty,
        "in."
    ))

    # ------------------------------------------------------------------
    # B: Baseline without leader
    # ------------------------------------------------------------------
    if df_ld is not None and "Self_L/D_error" in df_ld.columns:
        m, lo, hi = col_stats(df_ld, "Self_L/D_error")
        rows.append(make_row(
            "Baseline", "L/D",
            "Propagated from reference position (Y=0, Z=0) lift and drag",
            m, lo, hi, "nondim."
        ))

    if df_fc is not None:
        for label, col, units in [
            ("Velocity",       "Velocity Error (+/- m/s)",      "m/s"),
            (r"$C_{L}$",          "Lift Coefficient Error",         "nondim."),
            (r"$C_{D}$",          "Drag Coefficient Error",         "nondim."),
        ]:
            if col in df_fc.columns:
                m, lo, hi = col_stats(df_fc, col)
                rows.append(make_row(
                    "Baseline", label,
                    r"Averaged across all $\alpha$ values from polar runs",
                    m, lo, hi, units
                ))

    if df_bp is not None:
        tap_names = {f"tap{t}" for t in upper_surface_taps}
        bp_vals = df_bp[df_bp['tap'].isin(tap_names)]['self_data_std_true'].dropna().values
        if len(bp_vals) > 0:
            rows.append(make_row(
                "Baseline", "Pressure",
                f"Autocorrelation-corrected; pooled across {len(upper_surface_taps)} upper-surface taps",
                bp_vals.mean(), bp_vals.min(), bp_vals.max(),
                pressure_units
            ))

    # ------------------------------------------------------------------
    # C: Sting / strut support
    # ------------------------------------------------------------------
    if df_ld is not None:
        for label, col_unc in [
            ("Lift", "Sting Lift True Error"),
            ("Drag", "Sting Drag True Error"),
        ]:
            if col_unc in df_ld.columns:
                m, lo, hi = col_stats(df_ld, col_unc)
                rows.append(make_row(
                    "Sting/strut", label,
                    "Autocorrelation-corrected; propagated from sting-only run and sting bias",
                    m, lo, hi, force_units
                ))

    # ------------------------------------------------------------------
    # D: Formation-flight run with leader
    # ------------------------------------------------------------------
    if df_ld is not None:
        for label, col in [
            ("Lift force",  "Lift True Error"),
            ("Drag force",  "Drag True Error"),
        ]:
            if col in df_ld.columns:
                m, lo, hi = col_stats(df_ld, col)
                rows.append(make_row(
                    "Formation run", label,
                    "Autocorrelation-corrected; propagated from test and bias runs",
                    m, lo, hi, force_units
                ))

    if df_p is not None:
        tap_error_cols = [
            f"tap{t}_delta_pressure_error" for t in upper_surface_taps
            if f"tap{t}_delta_pressure_error" in df_p.columns
        ]
        if tap_error_cols:
            all_vals = df_p[tap_error_cols].values.ravel()
            all_vals = all_vals[~np.isnan(all_vals)]
            rows.append(make_row(
                "Formation run", r"$\Delta p$",
                f"Autocorrelation-corrected; pooled across {len(upper_surface_taps)} upper-surface taps",
                all_vals.mean(), all_vals.min(), all_vals.max(),
                pressure_units
            ))

        if "pitot_delta_pressure_error" in df_p.columns:
            m, lo, hi = col_stats(df_p, "pitot_delta_pressure_error")
            rows.append(make_row(
                "Formation run", "Pitot pressure",
                "Autocorrelation-corrected; propagated from pitot test and bias runs",
                m, lo, hi, pressure_units
            ))

    # ------------------------------------------------------------------
    # E: Derived / propagated uncertainty
    # ------------------------------------------------------------------
    if df_ld is not None:
        for label, col, source in [
            ("L/D",
             "L/D True Error",
             "Propagated from sting-corrected lift and drag via ratio rule"),
            (r"$\Delta$(L/D)",
             "delta_L_D_error",
             r"Propagated from formation L/D and baseline L/D"),
        ]:
            if col in df_ld.columns:
                m, lo, hi = col_stats(df_ld, col)
                rows.append(make_row(
                    "Derived", label,
                    source,
                    m, lo, hi, "nondim."
                ))

    return rows


def write_latex(rows, output_path):
    col_spec = r"l p{1.8cm} p{4.0cm} r p{1.8cm} l"
    header = (
        r"        \textbf{Category} & \textbf{Item} & \textbf{Method} & "
        r"\textbf{Mean} & \textbf{Range} & \textbf{Units} \\"
    )

    # group rows by category to compute multirow spans
    groups = []
    for cells in rows:
        cat = cells[0]
        if groups and groups[-1][0] == cat:
            groups[-1][1].append(cells[1:])
        else:
            groups.append((cat, [cells[1:]]))

    lines = [
        r"        \footnotesize",
        r"        \renewcommand{\arraystretch}{0.75}",
        r"        \begin{longtable}{" + col_spec + "}",
        r"        \caption{Measurement system uncertainty summary across all test configurations.}"
        r" \label{tab:uncertainty_summary} \\",
        r"        \hline",
        header,
        r"        \hline",
        r"        \endfirsthead",
        r"        \hline",
        header,
        r"        \hline",
        r"        \endhead",
        r"        \hline \multicolumn{6}{r}{\textit{Continued on next page}} \\",
        r"        \endfoot",
        r"        \hline",
        r"        \endlastfoot",
    ]

    for cat, group_rows in groups:
        n = len(group_rows)
        for i, cells in enumerate(group_rows):
            cat_cell = rf"\multirow{{{n}}}{{*}}{{{cat}}}" if i == 0 else ""
            escaped = [c.replace("_", r"\_") for c in cells]
            lines.append("        " + cat_cell + " & " + " & ".join(escaped) + r" \\")
            if i < n - 1:
                lines.append(r"        \cline{2-6}")
            else:
                lines.append(r"        \hline")

    lines.append(r"        \end{longtable}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved: {output_path}")


def build_short_table(df_ld, df_p):
    rows = []

    rows.append(make_spec_row(
        "Equipment", "Tap transducer",
        r"Omega PX165-001B; $\pm$1\% of $\pm$248.8\,Pa range",
        0.01 * 248.8, pressure_units
    ))
    rows.append(make_spec_row(
        "Equipment", r"$F_{x,y}$ (load cell)",
        "ATI Mini40; manufacturer precision",
        1 / 50, force_units
    ))
    rows.append(make_spec_row(
        "Positioning", r"$\Delta x,\, \Delta y,\, \Delta z$",
        "Half of ruler smallest division (1/16 in.)",
        (1 / 16) / 2, "in."
    ))

    if df_p is not None:
        tap_error_cols = [
            f"tap{t}_delta_pressure_error" for t in upper_surface_taps
            if f"tap{t}_delta_pressure_error" in df_p.columns
        ]
        if tap_error_cols:
            all_vals = df_p[tap_error_cols].values.ravel()
            all_vals = all_vals[~np.isnan(all_vals)]
            rows.append(make_row(
                "Measurement", r"$\Delta p$ (upper taps)",
                f"Autocorrelation-corrected; pooled across {len(upper_surface_taps)} taps",
                all_vals.mean(), all_vals.min(), all_vals.max(),
                pressure_units
            ))

    if df_ld is not None:
        if "L/D True Error" in df_ld.columns:
            m, lo, hi = col_stats(df_ld, "L/D True Error")
            rows.append(make_row(
                "Measurement", "L/D",
                "Autocorrelation-corrected; sting-corrected lift and drag",
                m, lo, hi, "nondim."
            ))
        if "delta_L_D_error" in df_ld.columns:
            m, lo, hi = col_stats(df_ld, "delta_L_D_error")
            rows.append(make_row(
                "Derived", r"$\Delta$(L/D)",
                "Propagated from formation and baseline L/D",
                m, lo, hi, "nondim."
            ))

    return rows


def write_short_latex(rows, output_path):
    col_spec = r"p{1.6cm} p{2.0cm} p{3.5cm} p{1.1cm} p{2.0cm} p{1.1cm}"
    header = (
        r"        \textbf{Category} & \textbf{Item} & \textbf{Method} & "
        r"\textbf{Mean} & \textbf{Range} & \textbf{Units} \\"
    )

    groups = []
    for cells in rows:
        cat = cells[0]
        if groups and groups[-1][0] == cat:
            groups[-1][1].append(cells[1:])
        else:
            groups.append((cat, [cells[1:]]))

    lines = [
        r"        \begin{table}[htbp]",
        r"        \centering",
        r"        \footnotesize",
        r"        \setlength{\tabcolsep}{4pt}",
        r"        \renewcommand{\arraystretch}{1.0}",
        r"        \caption{Key measurement uncertainties in the formation-flight experiment."
        r" Mean and range are computed across all test positions and angles of attack."
        r" See Table~\ref{tab:uncertainty_summary} in the Supplementary Information for a complete breakdown.}",
        r"        \label{tab:uncertainty_short}",
        r"        \begin{tabular}{" + col_spec + "}",
        r"        \hline",
        header,
        r"        \hline",
    ]

    for cat, group_rows in groups:
        n = len(group_rows)
        for i, cells in enumerate(group_rows):
            cat_cell = rf"\multirow{{{n}}}{{*}}{{{cat}}}" if i == 0 else ""
            escaped = [c.replace("_", r"\_") for c in cells]
            lines.append("        " + cat_cell + " & " + " & ".join(escaped) + r" \\")
            lines.append(r"        \hline")

    lines += [
        r"        \end{tabular}",
        r"        \end{table}",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    df_ld = load_concat(l_d_stats_paths)
    df_p  = load_concat(pressure_stats_paths)
    df_fc = pd.read_csv(force_coeff_path) if os.path.exists(force_coeff_path) else None
    df_bp = pd.read_csv(baseline_pressure_self_path) if os.path.exists(baseline_pressure_self_path) else None
    rows  = build_table(df_ld, df_p, df_fc, df_bp)
    write_latex(rows, output_tex)

    short_rows = build_short_table(df_ld, df_p)
    write_short_latex(short_rows, "Post_processed_stats_summarized/Uncertainty/Short_Uncertainty_Table_for_main.tex")
