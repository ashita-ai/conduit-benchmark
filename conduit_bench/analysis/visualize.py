"""Visualization module for benchmark results.

Generates interactive charts and reports using Plotly:
- Regret curves with 95% confidence interval bands
- Cost-quality scatter plots (Pareto frontier)
- Model selection heatmaps over time
- Convergence detection plots
- HTML reports with embedded interactive charts
"""

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from conduit_bench.analysis.enhanced_report import generate_enhanced_html_report


def plot_cost_curves(
    algorithms_data: dict[str, dict[str, Any]],
    benchmark_data: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    show_ci: bool = True,
) -> go.Figure:
    """Plot cumulative cost curves over time for multiple algorithms.

    Args:
        algorithms_data: Dict mapping algorithm names to their metrics data
        benchmark_data: Optional raw benchmark data with cumulative_cost histories
        output_path: Optional path to save figure (HTML)
        show_ci: Whether to show 95% confidence interval bands

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Try to get cost history from benchmark data if available
    if benchmark_data and "algorithms" in benchmark_data:
        for algo in benchmark_data["algorithms"]:
            algo_name = algo.get("algorithm_name", "unknown")
            cost_history = algo.get("cumulative_cost", [])

            if not cost_history or not isinstance(cost_history, list):
                continue

            queries = np.arange(1, len(cost_history) + 1)

            # Add main line
            fig.add_trace(
                go.Scatter(
                    x=queries,
                    y=cost_history,
                    mode="lines",
                    name=algo_name,
                    line=dict(width=2.5),
                    hovertemplate=f"{algo_name}<br>Query: %{{x}}<br>Cost: $%{{y:.4f}}<extra></extra>",
                )
            )

            # Add CI bands if requested (simplified - would need multiple runs for real CI)
            if show_ci and len(cost_history) > 10:
                # Estimate CI from local variance (simplified)
                window = min(50, len(cost_history) // 4)
                cost_array = np.array(cost_history)
                std_estimate = np.std(cost_array[-window:]) if window > 0 else 0

                ci_lower = np.maximum(0, cost_array - 1.96 * std_estimate)
                ci_upper = cost_array + 1.96 * std_estimate

                # Add upper bound
                fig.add_trace(
                    go.Scatter(
                        x=queries,
                        y=ci_upper,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # Add filled area for CI
                fig.add_trace(
                    go.Scatter(
                        x=queries,
                        y=ci_lower,
                        mode="lines",
                        line=dict(width=0),
                        fillcolor=f"rgba(0,0,0,0.1)",
                        fill="tonexty",
                        name=f"{algo_name} 95% CI",
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        title={
            "text": "Cumulative Cost Over Time<br><sub>(Lower is Better)</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Number of Queries",
        yaxis_title="Cumulative Cost ($)",
        hovermode="x unified",
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        height=600,
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def plot_cost_quality_scatter(
    algorithms_data: dict[str, dict[str, Any]],
    pareto_optimal: list[str] | None = None,
    output_path: str | Path | None = None,
) -> go.Figure:
    """Plot cost vs quality scatter with Pareto frontier highlighted.

    Args:
        algorithms_data: Dict mapping algorithm names to their metrics
        pareto_optimal: List of Pareto optimal algorithm names
        output_path: Optional path to save figure (HTML)

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

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
    non_pareto_indices = [i for i, p in enumerate(is_pareto) if not p]
    if non_pareto_indices:
        fig.add_trace(
            go.Scatter(
                x=[costs[i] for i in non_pareto_indices],
                y=[qualities[i] for i in non_pareto_indices],
                mode="markers+text",
                marker=dict(size=12, color="gray", line=dict(color="black", width=1)),
                text=[names[i] for i in non_pareto_indices],
                textposition="top center",
                name="Non-Pareto",
                hovertemplate="<b>%{text}</b><br>Cost: $%{x:.4f}<br>Quality: %{y:.3f}<extra></extra>",
            )
        )

    # Plot Pareto optimal points
    pareto_indices = [i for i, p in enumerate(is_pareto) if p]
    if pareto_indices:
        fig.add_trace(
            go.Scatter(
                x=[costs[i] for i in pareto_indices],
                y=[qualities[i] for i in pareto_indices],
                mode="markers+text",
                marker=dict(
                    size=18,
                    color="gold",
                    symbol="star",
                    line=dict(color="darkred", width=2),
                ),
                text=[names[i] for i in pareto_indices],
                textposition="top center",
                textfont=dict(size=12, color="darkred"),
                name="Pareto Optimal",
                hovertemplate="<b>%{text}</b><br>Cost: $%{x:.4f}<br>Quality: %{y:.3f}<extra></extra>",
            )
        )

    # Draw Pareto frontier line if we have Pareto optimal points
    if pareto_optimal and len(pareto_optimal) > 1:
        pareto_costs = [costs[i] for i in pareto_indices]
        pareto_qualities = [qualities[i] for i in pareto_indices]

        # Sort by cost
        sorted_pairs = sorted(zip(pareto_costs, pareto_qualities))
        frontier_costs, frontier_qualities = zip(*sorted_pairs)

        fig.add_trace(
            go.Scatter(
                x=frontier_costs,
                y=frontier_qualities,
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="Pareto Frontier",
                hoverinfo="skip",
            )
        )

    # Add target quality line
    fig.add_hline(
        y=0.8,
        line_dash="dot",
        line_color="green",
        opacity=0.4,
        annotation_text="Target Quality (0.8)",
        annotation_position="right",
    )

    fig.update_layout(
        title={
            "text": "Cost-Quality Trade-off<br><sub>(Best algorithms: low cost + high quality)</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Total Cost ($)",
        yaxis_title="Average Quality",
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        height=700,
        hovermode="closest",
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def plot_convergence_comparison(
    algorithms_data: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
) -> go.Figure:
    """Plot convergence points comparison across algorithms.

    Args:
        algorithms_data: Dict mapping algorithm names to their convergence data
        output_path: Optional path to save figure (HTML)

    Returns:
        Plotly Figure object
    """
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
    none_converged = len(valid_points) == 0

    # Create bar chart
    colors = ["green" if c else "red" for c in converged_flags]

    # Create hover text
    hover_texts = []
    for point, converged in zip(convergence_points, converged_flags):
        if point > 0:
            hover_texts.append(f"Query #{int(point)}")
        else:
            hover_texts.append("Never Converged")

    fig = go.Figure(
        data=[
            go.Bar(
                x=algo_names,
                y=convergence_points,
                marker_color=colors,
                marker_line=dict(color="black", width=1.5),
                text=hover_texts,
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>",
            )
        ]
    )

    # Adjust title based on whether points are identical
    if none_converged:
        title_text = (
            "Algorithm Convergence Speed<br>"
            "<sub>(No algorithms converged - dataset may be too small)</sub>"
        )
    elif all_identical and valid_points:
        title_text = (
            f"Algorithm Convergence Speed<br>"
            f"<sub>(All algorithms converged at query #{int(valid_points[0])})</sub>"
        )
    else:
        title_text = "Algorithm Convergence Speed<br><sub>(Lower is Faster)</sub>"

    # Set y-axis range: if none converged, set a fixed range for visibility
    yaxis_config = {
        "title": "Convergence Point (Query #)",
    }
    if none_converged:
        # Set a small fixed range when all values are 0 to make the chart visible
        yaxis_config["range"] = [0, 1]
        yaxis_config["showticklabels"] = False

    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "family": "Inter, Arial, sans-serif"},
        },
        yaxis=yaxis_config,
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        height=600,
        showlegend=False,
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def plot_quality_ranking(
    algorithms_data: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
) -> go.Figure:
    """Plot quality ranking with confidence intervals.

    Args:
        algorithms_data: Dict mapping algorithm names to their quality data
        output_path: Optional path to save figure (HTML)

    Returns:
        Plotly Figure object
    """
    # Sort algorithms by average quality
    sorted_algos = sorted(
        algorithms_data.items(), key=lambda x: x[1].get("average_quality", 0), reverse=True
    )

    algo_names = [name for name, _ in sorted_algos]
    qualities = [data.get("average_quality", 0) for _, data in sorted_algos]
    ci_lowers = [data.get("quality_ci_lower", q) for (_, data), q in zip(sorted_algos, qualities)]
    ci_uppers = [data.get("quality_ci_upper", q) for (_, data), q in zip(sorted_algos, qualities)]

    # Calculate error bars
    error_x_minus = [q - l for q, l in zip(qualities, ci_lowers)]
    error_x_plus = [u - q for q, u in zip(qualities, ci_uppers)]

    # Dynamic figure height based on number of algorithms
    num_algos = len(algorithms_data)
    fig_height = max(600, num_algos * 60 + 150)

    fig = go.Figure()

    # Add horizontal bars with error bars
    fig.add_trace(
        go.Bar(
            y=algo_names,
            x=qualities,
            orientation="h",
            marker=dict(color="skyblue", line=dict(color="navy", width=1.5)),
            error_x=dict(
                type="data",
                symmetric=False,
                array=error_x_plus,
                arrayminus=error_x_minus,
                color="black",
                thickness=2,
            ),
            text=[f"{q:.3f}" for q in qualities],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Quality: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Quality Ranking with 95% CI<br><sub>(Higher is Better)</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Average Quality Score",
        xaxis=dict(range=[0, 1.0]),
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        height=fig_height,
        showlegend=False,
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def plot_cost_efficiency(
    algorithms_data: dict[str, dict[str, Any]],
    pareto_optimal: list[str],
    output_path: str | Path | None = None,
) -> go.Figure:
    """Plot cost efficiency (quality/cost ratio) as bar chart.

    Args:
        algorithms_data: Dict mapping algorithm names to their metrics data
        pareto_optimal: List of Pareto optimal algorithm names
        output_path: Optional path to save figure (HTML)

    Returns:
        Plotly Figure object
    """
    # Calculate efficiency ratio (quality per dollar)
    efficiency_data = []
    for name, data in algorithms_data.items():
        quality = data.get("average_quality", 0)
        cost = data.get("total_cost", 0)
        if cost > 0:
            efficiency = quality / cost
            efficiency_data.append((name, efficiency, name in pareto_optimal))

    # Sort by efficiency (descending)
    efficiency_data.sort(key=lambda x: x[1], reverse=True)

    algo_names = [item[0] for item in efficiency_data]
    efficiencies = [item[1] for item in efficiency_data]
    colors = ["#28a745" if item[2] else "#6c757d" for item in efficiency_data]

    # Dynamic figure height
    num_algos = len(efficiency_data)
    fig_height = max(600, num_algos * 60 + 150)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=algo_names,
            x=efficiencies,
            orientation="h",
            marker=dict(color=colors, line=dict(color="black", width=1.5)),
            text=[f"{e:.1f}" for e in efficiencies],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Efficiency: %{x:.2f} quality/$<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Cost Efficiency Comparison<br><sub>Quality Score per Dollar (Higher is Better)</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "family": "Inter, Arial, sans-serif"},
        },
        xaxis_title="Efficiency Ratio (Quality / Cost)",
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        height=fig_height,
        showlegend=False,
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def generate_html_report(
    analysis: dict[str, Any],
    output_dir: str | Path,
    chart_figs: dict[str, go.Figure] | None = None,
) -> Path:
    """Generate comprehensive HTML report with embedded interactive charts.

    Uses the enhanced report generator with beautiful gradient styling.

    Args:
        analysis: Complete analysis dictionary from metrics module
        output_dir: Directory to save HTML report
        chart_figs: Optional dict of chart type -> Plotly Figure mappings

    Returns:
        Path to generated HTML report
    """
    # Delegate to the enhanced report generator
    return generate_enhanced_html_report(analysis, output_dir, chart_figs)



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
    chart_figs = {}

    # Extract algorithm data
    algorithms = analysis.get("algorithms", {})

    # 1. Cost curves (skip if no time series data)
    if benchmark_data:
        cost_path = output_dir / "cost_curves.html"
        cost_fig = plot_cost_curves(algorithms, benchmark_data, output_path=cost_path)
        chart_paths["Cumulative Cost Curves"] = cost_path
        chart_figs["Cumulative Cost Curves"] = cost_fig

    # 2. Cost-quality scatter
    pareto_optimal = analysis.get("pareto_frontier", [])
    cost_quality_path = output_dir / "cost_quality_scatter.html"
    cost_quality_fig = plot_cost_quality_scatter(
        algorithms, pareto_optimal, output_path=cost_quality_path
    )
    chart_paths["Cost-Quality Trade-off"] = cost_quality_path
    chart_figs["Cost-Quality Trade-off"] = cost_quality_fig

    # 3. Convergence comparison
    convergence_path = output_dir / "convergence_comparison.html"
    convergence_fig = plot_convergence_comparison(algorithms, output_path=convergence_path)
    chart_paths["Convergence Speed"] = convergence_path
    chart_figs["Convergence Speed"] = convergence_fig

    # 4. Quality ranking
    quality_path = output_dir / "quality_ranking.html"
    quality_fig = plot_quality_ranking(algorithms, output_path=quality_path)
    chart_paths["Quality Ranking"] = quality_path
    chart_figs["Quality Ranking"] = quality_fig

    # 5. Cost efficiency
    efficiency_path = output_dir / "cost_efficiency.html"
    efficiency_fig = plot_cost_efficiency(algorithms, pareto_optimal, output_path=efficiency_path)
    chart_paths["Cost Efficiency"] = efficiency_path
    chart_figs["Cost Efficiency"] = efficiency_fig

    # 6. HTML report with embedded interactive charts
    html_path = generate_html_report(analysis, output_dir, chart_figs)
    chart_paths["HTML Report"] = html_path

    return chart_paths
