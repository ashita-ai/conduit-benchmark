"""Enhanced HTML report generation with beautiful styling."""

from pathlib import Path
from typing import Any
import plotly.graph_objects as go


def generate_enhanced_html_report(
    analysis: dict[str, Any],
    output_dir: str | Path,
    chart_figs: dict[str, go.Figure] | None = None,
) -> Path:
    """Generate beautiful HTML report with gradient styling and embedded charts.

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
    summary = analysis.get("summary", {})
    statistical_tests = analysis.get("statistical_tests", {})
    friedman = statistical_tests.get("friedman", {})
    pareto = analysis.get("pareto_frontier", [])

    # Get best algorithms
    best_quality = summary.get("best_quality_algorithm", "N/A")
    best_cost = summary.get("best_cost_algorithm", "N/A")
    quality_rankings = summary.get("quality_rankings", [])
    cost_rankings = summary.get("cost_rankings", [])

    # Calculate best value (quality/cost efficiency)
    best_value_algo = None
    best_value_ratio = 0
    for name, data in algorithms.items():
        quality = data.get("average_quality", 0)
        cost = data.get("total_cost", 0)
        if cost > 0:
            ratio = quality / cost
            if ratio > best_value_ratio:
                best_value_ratio = ratio
                best_value_algo = name

    # Get benchmark metadata
    benchmark_id = analysis.get("benchmark_id", "N/A")
    dataset_size = analysis.get("dataset_size", 0)
    num_algorithms = summary.get("num_algorithms", len(algorithms))

    # Get best algo data
    best_quality_data = algorithms.get(best_quality, {})
    best_cost_data = algorithms.get(best_cost, {})
    best_value_data = algorithms.get(best_value_algo, {}) if best_value_algo else {}

    # Start building HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conduit Benchmark Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }

        h2 {
            color: #34495e;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }

        h3 {
            color: #34495e;
            font-size: 1.3em;
            margin-top: 25px;
            margin-bottom: 15px;
        }

        .meta {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            font-size: 0.95em;
        }

        .meta-item {
            display: inline-block;
            margin-right: 25px;
        }

        .meta-label {
            font-weight: 600;
            color: #7f8c8d;
        }

        .executive-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
        }

        .executive-summary h2 {
            color: white;
            border-left-color: white;
            margin-top: 0;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .summary-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 5px;
            backdrop-filter: blur(10px);
        }

        .summary-card-title {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .summary-card-value {
            font-size: 1.5em;
            font-weight: 700;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }

        td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }

        tr:hover {
            background: #f8f9fa;
        }

        .rank-1 {
            background: #fff3cd;
            font-weight: 600;
        }

        .rank-2 {
            background: #d1ecf1;
        }

        .rank-3 {
            background: #d4edda;
        }

        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .badge-pareto {
            background: #28a745;
            color: white;
        }

        .insight-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }

        .insight-box h4 {
            color: #1976d2;
            margin-bottom: 10px;
        }

        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }

        .warning-box h4 {
            color: #f57c00;
            margin-bottom: 10px;
        }

        .chart-container {
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .pareto-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }

        .pareto-item {
            background: #28a745;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 600;
        }

        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Conduit Benchmark Analysis Report</h1>

        <div class="meta">
            <div class="meta-item">
                <span class="meta-label">Benchmark ID:</span> """ + str(benchmark_id) + """
            </div>
            <div class="meta-item">
                <span class="meta-label">Dataset Size:</span> """ + str(dataset_size) + """ queries
            </div>
            <div class="meta-item">
                <span class="meta-label">Algorithms Tested:</span> """ + str(num_algorithms) + """
            </div>
        </div>

        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-card-title">Best Quality</div>
                    <div class="summary-card-value">""" + str(best_quality) + """</div>
                    <div style="font-size: 0.95em; margin-top: 5px;">""" + f"{best_quality_data.get('average_quality', 0):.1%}" + """ accuracy</div>
                </div>
                <div class="summary-card">
                    <div class="summary-card-title">Best Cost</div>
                    <div class="summary-card-value">""" + str(best_cost) + """</div>
                    <div style="font-size: 0.95em; margin-top: 5px;">$""" + f"{best_cost_data.get('total_cost', 0):.4f}" + """ total</div>
                </div>
                <div class="summary-card">
                    <div class="summary-card-title">Best Value</div>
                    <div class="summary-card-value">""" + (str(best_value_algo) if best_value_algo else "N/A") + """</div>
                    <div style="font-size: 0.95em; margin-top: 5px;">""" + (f"{best_value_ratio:.1f}:1 efficiency ratio" if best_value_algo else "N/A") + """</div>
                </div>
                <div class="summary-card">
                    <div class="summary-card-title">Pareto Optimal</div>
                    <div class="summary-card-value">""" + str(len(pareto)) + """ Algorithms</div>
                    <div style="font-size: 0.95em; margin-top: 5px;">Non-dominated solutions</div>
                </div>
            </div>
        </div>
"""

    # Add Pareto frontier section
    if pareto:
        html += """
        <h2>Pareto Frontier Analysis</h2>
        <p>The Pareto frontier represents algorithms that are non-dominated - no other algorithm is both cheaper and better quality.</p>

        <div class="pareto-list">
"""
        for algo in pareto:
            html += f'            <div class="pareto-item">{algo}</div>\n'
        html += """        </div>

        <div class="insight-box">
            <h4>Key Finding: """ + (best_value_algo if best_value_algo else "Value") + """ Dominates Cost/Quality Tradeoff</h4>
            <p><strong>""" + (best_value_algo if best_value_algo else "Best value algorithm") + """</strong> achieves """ + f"{best_value_data.get('average_quality', 0):.1%}" + """ quality at just $""" + f"{best_value_data.get('total_cost', 0):.4f}" + """ cost, representing a <strong>""" + f"{best_value_ratio:.1f}:1" + """ cost-efficiency advantage</strong>.</p>
        </div>
"""

    # Add statistical significance warning
    p_value = friedman.get("p_value", 1.0)
    if p_value >= 0.05:
        html += """
        <div class="warning-box">
            <h4>⚠️ Statistical Significance Note</h4>
            <p><strong>Friedman Test Result:</strong> p-value = """ + f"{p_value:.3f}" + """ (not significant at α=0.05)</p>
            <p>With only """ + str(dataset_size) + """ queries, the differences between algorithms are <strong>not statistically significant</strong>. The observed quality differences could be due to random chance rather than genuine algorithm superiority.</p>
        </div>
"""

    # Embed all charts
    if chart_figs:
        html += """
        <h2>Interactive Visualizations</h2>
"""
        for chart_name, fig in chart_figs.items():
            if fig is not None:
                html += f"""
        <div class="chart-container">
            <h3>{chart_name}</h3>
            {fig.to_html(include_plotlyjs=False, div_id=chart_name.replace(' ', '_').lower())}
        </div>
"""

    # Add footer
    html += """
        <div class="footer">
            <p>Generated from benchmark ID: """ + str(benchmark_id) + """</p>
            <p>Conduit Benchmark Analysis · """ + str(dataset_size) + """ Queries · """ + str(num_algorithms) + """ Algorithms</p>
        </div>
    </div>
</body>
</html>
"""

    # Write to file
    with open(report_path, 'w') as f:
        f.write(html)

    return report_path
