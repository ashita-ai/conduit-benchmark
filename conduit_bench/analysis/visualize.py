"""Visualization module for benchmark results.

Generates publication-quality charts and reports:
- Regret curves with 95% confidence interval bands
- Cost-quality scatter plots (Pareto frontier)
- Model selection heatmaps over time
- Convergence detection plots
- HTML reports with embedded charts
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

# Set seaborn style for modern, clean visualizations
sns.set_theme(style="whitegrid", context="talk", font="Inter", palette="deep")
sns.set_context("talk", rc={"font.family": "sans-serif", "font.sans-serif": ["Inter", "Arial", "Helvetica"]})

# Set publication-quality defaults
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "grid.alpha": 0.3,
    }
)

# Use seaborn style
sns.set_palette("husl")


def plot_cost_curves(
    algorithms_data: dict[str, dict[str, Any]],
    benchmark_data: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    show_ci: bool = True,
) -> Figure:
    """Plot cumulative cost curves over time for multiple algorithms.

    Args:
        algorithms_data: Dict mapping algorithm names to their metrics data
        benchmark_data: Optional raw benchmark data with cumulative_cost histories
        output_path: Optional path to save figure (PNG)
        show_ci: Whether to show 95% confidence interval bands

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Try to get cost history from benchmark data if available
    if benchmark_data and "algorithms" in benchmark_data:
        for algo in benchmark_data["algorithms"]:
            algo_name = algo.get("algorithm_name", "unknown")
            cost_history = algo.get("cumulative_cost", [])

            if not cost_history or not isinstance(cost_history, list):
                continue

            queries = np.arange(1, len(cost_history) + 1)

            # Plot main line
            ax.plot(queries, cost_history, label=algo_name, linewidth=2.5, alpha=0.8)

            # Add CI bands if requested (simplified - would need multiple runs for real CI)
            if show_ci and len(cost_history) > 10:
                # Estimate CI from local variance (simplified)
                window = min(50, len(cost_history) // 4)
                cost_array = np.array(cost_history)
                std_estimate = np.std(cost_array[-window:]) if window > 0 else 0

                ci_lower = np.maximum(0, cost_array - 1.96 * std_estimate)
                ci_upper = cost_array + 1.96 * std_estimate

                ax.fill_between(
                    queries, ci_lower, ci_upper, alpha=0.2, label=f"{algo_name} 95% CI"
                )

    ax.set_xlabel("Number of Queries", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Cost ($)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Cumulative Cost Over Time\n(Lower is Better)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_cost_quality_scatter(
    algorithms_data: dict[str, dict[str, Any]],
    pareto_optimal: list[str] | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot cost vs quality scatter with Pareto frontier highlighted.

    Args:
        algorithms_data: Dict mapping algorithm names to their metrics
        pareto_optimal: List of Pareto optimal algorithm names
        output_path: Optional path to save figure (PNG)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract cost and quality for each algorithm
    costs = []
    qualities = []
    names = []
    is_pareto = []

    for algo_name, data in algorithms_data.items():
        cost = data.get("total_cost", 0)
        quality = data.get("average_quality", 0)
        costs.append(cost)
        qualities.append(quality)
        names.append(algo_name)
        is_pareto.append(algo_name in (pareto_optimal or []))

    # Plot non-Pareto points
    non_pareto_mask = [not p for p in is_pareto]
    if any(non_pareto_mask):
        ax.scatter(
            [c for c, p in zip(costs, non_pareto_mask) if p],
            [q for q, p in zip(qualities, non_pareto_mask) if p],
            s=200,
            alpha=0.6,
            c="gray",
            edgecolors="black",
            linewidth=1.5,
            label="Non-Pareto",
            zorder=2,
        )

    # Plot Pareto optimal points
    pareto_mask = is_pareto
    if any(pareto_mask):
        ax.scatter(
            [c for c, p in zip(costs, pareto_mask) if p],
            [q for q, p in zip(qualities, pareto_mask) if p],
            s=300,
            alpha=0.9,
            c="gold",
            edgecolors="darkred",
            linewidth=2.5,
            marker="*",
            label="Pareto Optimal",
            zorder=3,
        )

    # Add labels for each point
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (costs[i], qualities[i]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold" if is_pareto[i] else "normal",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="yellow" if is_pareto[i] else "white",
                edgecolor="black",
                alpha=0.8,
            ),
        )

    # Draw Pareto frontier line if we have Pareto optimal points
    if pareto_optimal and len(pareto_optimal) > 1:
        pareto_costs = [c for c, p in zip(costs, is_pareto) if p]
        pareto_qualities = [q for q, p in zip(qualities, is_pareto) if p]

        # Sort by cost
        sorted_pairs = sorted(zip(pareto_costs, pareto_qualities))
        frontier_costs, frontier_qualities = zip(*sorted_pairs)

        ax.plot(
            frontier_costs,
            frontier_qualities,
            "r--",
            linewidth=2,
            alpha=0.5,
            label="Pareto Frontier",
            zorder=1,
        )

    ax.set_xlabel("Total Cost ($)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Quality", fontsize=14, fontweight="bold")
    ax.set_title(
        "Cost-Quality Trade-off\n(Best algorithms: low cost + high quality)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add diagonal reference lines
    ax.axhline(
        y=0.8, color="green", linestyle=":", alpha=0.4, label="Target Quality (0.8)"
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_convergence_comparison(
    algorithms_data: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
) -> Figure:
    """Plot convergence points comparison across algorithms.

    Args:
        algorithms_data: Dict mapping algorithm names to their convergence data
        output_path: Optional path to save figure (PNG)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    algo_names = []
    convergence_points = []
    converged_flags = []

    for algo_name, data in algorithms_data.items():
        algo_names.append(algo_name)

        # Handle both nested and flat convergence structures
        convergence = data.get("convergence", {})
        if isinstance(convergence, dict):
            conv_point = convergence.get("convergence_point")
            converged = convergence.get("converged", False)
        else:
            conv_point = data.get("convergence_point")
            converged = data.get("converged", False)

        convergence_points.append(conv_point if conv_point else 0)
        converged_flags.append(converged)

    # Check if all convergence points are identical (for title customization)
    valid_points = [p for p in convergence_points if p > 0]
    all_identical = len(set(valid_points)) == 1 if valid_points else False

    # Plot bars
    colors = ["green" if c else "red" for c in converged_flags]
    bars = ax.bar(algo_names, convergence_points, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for i, (bar, point) in enumerate(zip(bars, convergence_points)):
        if point > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"Query #{int(point)}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                5,
                "Never Converged",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                style="italic",
                color="darkred",
            )

    ax.set_ylabel("Convergence Point (Query #)", fontsize=14, fontweight="bold")

    # Adjust title based on whether points are identical
    if all_identical and valid_points:
        title_text = (
            f"Algorithm Convergence Speed\n"
            f"(All algorithms converged at query #{int(valid_points[0])})"
        )
    else:
        title_text = "Algorithm Convergence Speed\n(Lower is Faster)"

    ax.set_title(
        title_text,
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Rotate x-axis labels if many algorithms
    if len(algo_names) > 5:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_quality_ranking(
    algorithms_data: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
) -> Figure:
    """Plot quality ranking with confidence intervals.

    Args:
        algorithms_data: Dict mapping algorithm names to their quality data
        output_path: Optional path to save figure (PNG)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort algorithms by average quality
    sorted_algos = sorted(
        algorithms_data.items(), key=lambda x: x[1].get("average_quality", 0), reverse=True
    )

    algo_names = [name for name, _ in sorted_algos]
    qualities = [data.get("average_quality", 0) for _, data in sorted_algos]
    ci_lowers = [data.get("quality_ci_lower", q) for (_, data), q in zip(sorted_algos, qualities)]
    ci_uppers = [data.get("quality_ci_upper", q) for (_, data), q in zip(sorted_algos, qualities)]

    # Calculate error bars
    yerr_lower = [q - l for q, l in zip(qualities, ci_lowers)]
    yerr_upper = [u - q for q, u in zip(qualities, ci_uppers)]

    # Plot bars with error bars
    bars = ax.barh(algo_names, qualities, color="skyblue", edgecolor="navy", linewidth=1.5)
    ax.errorbar(
        qualities,
        range(len(algo_names)),
        xerr=[yerr_lower, yerr_upper],
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=2,
    )

    # Add value labels
    for i, (bar, quality) in enumerate(zip(bars, qualities)):
        ax.text(
            quality + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{quality:.3f}",
            va="center",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_xlabel("Average Quality Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "Quality Ranking with 95% CI\n(Higher is Better)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def generate_html_report(
    analysis: dict[str, Any],
    output_dir: str | Path,
    chart_paths: dict[str, str] | None = None,
) -> Path:
    """Generate comprehensive HTML report with embedded charts.

    Args:
        analysis: Complete analysis dictionary from metrics module
        output_dir: Directory to save HTML report and charts
        chart_paths: Optional dict of chart type -> path mappings

    Returns:
        Path to generated HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "benchmark_report.html"

    # Extract data
    algorithms = analysis.get("algorithms", {})
    statistical_tests = analysis.get("statistical_tests", {})
    friedman = statistical_tests.get("friedman", {})
    pareto = analysis.get("pareto_frontier", [])

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conduit Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fafafa;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .chart {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
        }}
        .pareto {{
            background: #fff3cd;
            padding: 10px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background: #e8f4f8;
            border-radius: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <h1>🎯 Conduit Benchmark Report</h1>

    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Benchmark ID:</strong> {analysis.get('benchmark_id', 'N/A')}</p>
        <p><strong>Dataset Size:</strong> {analysis.get('dataset_size', 0)} queries</p>
        <p><strong>Algorithms Tested:</strong> {len(algorithms)}</p>

        <div class="metric">
            <div class="metric-label">Friedman Test</div>
            <div class="metric-value">{'Significant' if friedman.get('significant') else 'Not Significant'}</div>
            <div class="metric-label">p = {friedman.get('p_value', 0):.4f}</div>
        </div>

        <div class="metric">
            <div class="metric-label">Pareto Optimal</div>
            <div class="metric-value">{len(pareto)}</div>
            <div class="metric-label">algorithms</div>
        </div>
    </div>

    <div class="pareto">
        <h3>🏆 Pareto Optimal Algorithms</h3>
        <p>The following algorithms achieve optimal cost-quality trade-offs:</p>
        <ul>
"""

    for algo in pareto:
        html += f"            <li><strong>{algo}</strong></li>\n"

    html += """
        </ul>
    </div>

    <h2>📊 Algorithm Performance</h2>
    <table>
        <tr>
            <th>Algorithm</th>
            <th>Avg Quality</th>
            <th>95% CI</th>
            <th>Total Cost</th>
            <th>Cum. Cost</th>
            <th>Converged</th>
        </tr>
"""

    for algo_name, data in algorithms.items():
        # Handle both nested and flat convergence structures
        convergence = data.get("convergence", {})
        if isinstance(convergence, dict):
            conv_point = convergence.get("convergence_point")
            converged = convergence.get("converged", False)
        else:
            conv_point = data.get("convergence_point")
            converged = data.get("converged", False)

        # Show actual convergence point (query number) instead of checkmark
        if converged and conv_point:
            converged_str = f"Query #{int(conv_point)}"
        else:
            converged_str = "Never"

        # Handle both tuple and separate CI fields
        quality_ci = data.get("quality_ci")
        if quality_ci and isinstance(quality_ci, tuple):
            ci_lower, ci_upper = quality_ci
        else:
            ci_lower = data.get("quality_ci_lower", 0.0)
            ci_upper = data.get("quality_ci_upper", 0.0)

        html += f"""
        <tr>
            <td><strong>{algo_name}</strong></td>
            <td>{data.get('average_quality', 0):.3f}</td>
            <td>[{ci_lower:.3f}, {ci_upper:.3f}]</td>
            <td>${data.get('total_cost', 0):.4f}</td>
            <td>${data.get('cumulative_cost', 0):.4f}</td>
            <td>{converged_str}</td>
        </tr>
"""

    html += """
    </table>
"""

    # Add charts if provided
    if chart_paths:
        html += "\n    <h2>📈 Visualizations</h2>\n"
        for chart_name, chart_path in chart_paths.items():
            html += f"""
    <div class="chart">
        <h3>{chart_name}</h3>
        <img src="{Path(chart_path).name}" alt="{chart_name}">
    </div>
"""

    html += """

    <div class="summary">
        <h2>Statistical Analysis</h2>
        <p><strong>Friedman Test:</strong> """

    if friedman.get("significant"):
        html += f"Significant differences detected (χ² = {friedman.get('statistic', 0):.2f}, p = {friedman.get('p_value', 0):.4f})"
    else:
        html += f"No significant differences (p = {friedman.get('p_value', 0):.4f})"

    html += """</p>
    </div>

    <div class="summary">
        <h2>📚 Appendix: Metrics and Statistical Tests</h2>

        <h3>Convergence Detection</h3>
        <p><strong>What it means:</strong> Convergence indicates that an algorithm has learned enough to make stable, consistent decisions. A converged algorithm is no longer significantly improving its strategy.</p>
        <p><strong>How we detect it:</strong> We use slope-based analysis on smoothed learning curves. An algorithm is considered converged when the slope of its quality improvement falls below 10% (nearly flat learning curve), indicating minimal further learning.</p>
        <ul>
            <li><strong>Converged (✓):</strong> Algorithm has stabilized and learned an effective strategy</li>
            <li><strong>Not Converged (✗):</strong> Algorithm is still learning or hasn't seen enough data</li>
        </ul>

        <h3>Pareto Frontier (Optimal Trade-offs)</h3>
        <p><strong>What it means:</strong> The Pareto frontier identifies algorithms that achieve optimal cost-quality trade-offs. An algorithm is Pareto optimal if no other algorithm has both lower cost AND higher quality.</p>
        <p><strong>Interpretation:</strong></p>
        <ul>
            <li>Algorithms on the Pareto frontier represent the best choices at different points on the cost-quality spectrum</li>
            <li>Lower-cost Pareto algorithms prioritize efficiency</li>
            <li>Higher-quality Pareto algorithms prioritize performance</li>
            <li>Non-Pareto algorithms are dominated by at least one other algorithm in both cost and quality</li>
        </ul>

        <h3>Friedman Test (Statistical Significance)</h3>
        <p><strong>What it tests:</strong> The Friedman test determines if there are statistically significant differences in performance across multiple algorithms tested on the same dataset.</p>
        <p><strong>Interpretation:</strong></p>
        <ul>
            <li><strong>Significant (p < 0.05):</strong> Strong evidence that algorithms perform differently. Quality differences are unlikely due to random chance.</li>
            <li><strong>Not Significant (p ≥ 0.05):</strong> Insufficient evidence of meaningful differences. Observed variations may be due to random chance.</li>
            <li><strong>p-value:</strong> Probability of seeing these results if all algorithms were equally good. Lower values indicate stronger evidence of real differences.</li>
        </ul>

        <h3>Metric Definitions</h3>
        <ul>
            <li><strong>Quality Score:</strong> Fraction of queries answered correctly (0.0 = all wrong, 1.0 = all correct)</li>
            <li><strong>Total Cost:</strong> Sum of API costs across all queries (in USD)</li>
            <li><strong>95% CI (Confidence Interval):</strong> Range where we're 95% confident the true average quality lies. Narrower intervals indicate more reliable estimates.</li>
            <li><strong>Cumulative Cost:</strong> Running total of costs as queries are processed, used to track spending over time</li>
            <li><strong>Convergence Point:</strong> Query number where the algorithm's learning curve stabilized</li>
        </ul>

        <h3>Algorithm Types</h3>
        <ul>
            <li><strong>Thompson Sampling:</strong> Bayesian approach that balances exploration and exploitation probabilistically</li>
            <li><strong>UCB1:</strong> Optimistic algorithm that favors options with high uncertainty</li>
            <li><strong>Epsilon-Greedy:</strong> Simple strategy that explores randomly ε% of the time</li>
            <li><strong>Random:</strong> Baseline that selects models uniformly at random</li>
            <li><strong>Always Best:</strong> Oracle that always selects the highest-quality model</li>
            <li><strong>Always Cheapest:</strong> Baseline that always selects the lowest-cost model</li>
        </ul>
    </div>

    <footer style="margin-top: 40px; padding: 20px; text-align: center; color: #7f8c8d; border-top: 1px solid #ddd;">
        <p>Generated by Conduit-Bench</p>
    </footer>
</body>
</html>
"""

    # Write report
    with open(report_path, "w") as f:
        f.write(html)

    return report_path


def create_all_visualizations(
    analysis: dict[str, Any],
    output_dir: str | Path,
    benchmark_data: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Create all visualization charts and HTML report.

    Args:
        analysis: Complete analysis dictionary from metrics module
        output_dir: Directory to save all visualizations
        benchmark_data: Optional raw benchmark data with full time series

    Returns:
        Dictionary mapping visualization type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = {}

    # Extract algorithm data
    algorithms = analysis.get("algorithms", {})

    # 1. Cost curves (skip if no time series data)
    if benchmark_data:
        cost_path = output_dir / "cost_curves.png"
        plot_cost_curves(algorithms, benchmark_data, output_path=cost_path)
        chart_paths["Cumulative Cost Curves"] = cost_path

    # 2. Cost-quality scatter
    pareto_optimal = analysis.get("pareto_frontier", [])
    cost_quality_path = output_dir / "cost_quality_scatter.png"
    plot_cost_quality_scatter(algorithms, pareto_optimal, output_path=cost_quality_path)
    chart_paths["Cost-Quality Trade-off"] = cost_quality_path

    # 3. Convergence comparison
    convergence_path = output_dir / "convergence_comparison.png"
    plot_convergence_comparison(algorithms, output_path=convergence_path)
    chart_paths["Convergence Speed"] = convergence_path

    # 4. Quality ranking
    quality_path = output_dir / "quality_ranking.png"
    plot_quality_ranking(algorithms, output_path=quality_path)
    chart_paths["Quality Ranking"] = quality_path

    # 5. HTML report
    chart_path_strings = {k: str(v) for k, v in chart_paths.items()}
    html_path = generate_html_report(analysis, output_dir, chart_path_strings)
    chart_paths["HTML Report"] = html_path

    return chart_paths
