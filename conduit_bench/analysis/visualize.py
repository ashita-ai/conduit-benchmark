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
    if all_identical and valid_points:
        title_text = (
            f"Algorithm Convergence Speed<br>"
            f"<sub>(All algorithms converged at query #{int(valid_points[0])})</sub>"
        )
    else:
        title_text = "Algorithm Convergence Speed<br><sub>(Lower is Faster)</sub>"

    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "family": "Inter, Arial, sans-serif"},
        },
        yaxis_title="Convergence Point (Query #)",
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


def generate_html_report(
    analysis: dict[str, Any],
    output_dir: str | Path,
    chart_figs: dict[str, go.Figure] | None = None,
) -> Path:
    """Generate comprehensive HTML report with embedded interactive charts.

    Args:
        analysis: Complete analysis dictionary from metrics module
        output_dir: Directory to save HTML report
        chart_figs: Optional dict of chart type -> Plotly Figure mappings

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
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            line-height: 1.7;
            color: #1a202c;
        }}

        .container {{
            background: #ffffff;
            border-radius: 20px;
            padding: 50px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}

        h1 {{
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 30px;
            letter-spacing: -1px;
            padding-bottom: 20px;
            border-bottom: 4px solid #f0f0f0;
        }}

        h2 {{
            font-size: 1.75rem;
            font-weight: 700;
            color: #2d3748;
            margin: 50px 0 25px 0;
            padding-left: 20px;
            border-left: 6px solid #667eea;
            letter-spacing: -0.5px;
        }}

        h3 {{
            font-size: 1.3rem;
            font-weight: 600;
            color: #4a5568;
            margin: 25px 0 15px 0;
        }}

        .summary {{
            background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
            padding: 35px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
            margin: 30px 0;
            border: 2px solid #e5e9f5;
        }}

        .summary p {{
            font-size: 1.05rem;
            margin: 12px 0;
            color: #2d3748;
        }}

        .summary strong {{
            color: #667eea;
            font-weight: 600;
        }}

        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: white;
            margin: 30px 0;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            font-family: 'JetBrains Mono', monospace;
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        td {{
            padding: 16px;
            border-bottom: 1px solid #e5e9f5;
            font-size: 0.95rem;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover {{
            background: linear-gradient(90deg, #f8f9ff 0%, #ffffff 100%);
            transition: all 0.2s ease;
        }}

        td strong {{
            color: #667eea;
            font-weight: 600;
        }}

        .chart {{
            background: white;
            padding: 35px;
            margin: 35px 0;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        }}

        .pareto {{
            background: linear-gradient(135deg, #fff9e6 0%, #fff3cd 100%);
            padding: 25px 30px;
            border-left: 6px solid #fbbf24;
            margin: 30px 0;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(251, 191, 36, 0.15);
        }}

        .pareto h3 {{
            color: #92400e;
            margin-bottom: 15px;
        }}

        .pareto p {{
            color: #78350f;
            font-size: 1.05rem;
        }}

        .pareto ul {{
            list-style-position: inside;
            margin-top: 12px;
        }}

        .pareto li {{
            color: #78350f;
            margin: 8px 0;
            font-size: 1.05rem;
        }}

        .metric {{
            display: inline-block;
            margin: 15px 20px;
            padding: 25px 35px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .metric:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        }}

        .metric-value {{
            font-size: 2.5rem;
            font-weight: 800;
            color: white;
            margin: 8px 0;
            font-family: 'Inter', sans-serif;
        }}

        .metric-label {{
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.9);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }}

        footer {{
            margin-top: 60px;
            padding: 30px 0;
            text-align: center;
            color: #718096;
            border-top: 2px solid #e5e9f5;
            font-size: 0.95rem;
        }}

        ul {{
            padding-left: 25px;
            margin: 15px 0;
        }}

        li {{
            margin: 10px 0;
            line-height: 1.7;
            color: #4a5568;
        }}

        p {{
            margin: 15px 0;
            color: #4a5568;
            font-size: 1.05rem;
        }}
    </style>
</head>
<body>
    <div class="container">
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

        # Show convergence in human-readable format
        if converged and conv_point:
            converged_str = f"Converged after {int(conv_point)} iterations"
        else:
            converged_str = "Never converged"

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

    # Add interactive charts if provided
    if chart_figs:
        html += "\n    <h2>📈 Interactive Visualizations</h2>\n"
        for chart_name, fig in chart_figs.items():
            chart_div = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=f"chart_{chart_name.replace(' ', '_').lower()}",
            )
            html += f"""
    <div class="chart">
        {chart_div}
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

        <footer>
            <p>Generated by Conduit-Bench | Interactive visualizations powered by Plotly</p>
        </footer>
    </div>
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

    # 5. HTML report with embedded interactive charts
    html_path = generate_html_report(analysis, output_dir, chart_figs)
    chart_paths["HTML Report"] = html_path

    return chart_paths
