import os
import pandas as pd
from datetime import datetime

# --- CDN assets ---
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
BOOTSTRAP_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"
DATATABLES_CSS = "https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css"
DATATABLES_JS = "https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"
DATATABLES_BOOTSTRAP_JS = "https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"
JQUERY_JS = "https://code.jquery.com/jquery-3.7.0.min.js"


def save_html_report(run_id: str, trades_df: pd.DataFrame, output_path: str):
    """
    Generate a Bootstrap + DataTables styled HTML report with color-coded trade actions.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = trades_df.copy()
    if "Action" in df.columns:
        # Add color-coded HTML tags
        color_map = {
            "BUY": "badge bg-success",
            "SELL": "badge bg-danger",
            "HOLD": "badge bg-secondary",
        }
        df["Action"] = df["Action"].apply(
            lambda a: f'<span class="{color_map.get(a, "badge bg-secondary")}">{a}</span>'
        )

    html_table = df.to_html(
        index=False,
        escape=False,
        classes="table table-striped table-bordered table-hover align-middle",
    )

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NASDAQ Scan Report {run_id}</title>
<link rel="stylesheet" href="{BOOTSTRAP_CSS}">
<link rel="stylesheet" href="{DATATABLES_CSS}">
<script src="{JQUERY_JS}"></script>
<script src="{BOOTSTRAP_JS}"></script>
<script src="{DATATABLES_JS}"></script>
<script src="{DATATABLES_BOOTSTRAP_JS}"></script>
<style>
body {{
  background-color: #f8f9fa;
  padding: 2rem;
}}
h1 {{
  margin-bottom: 1rem;
}}
.table-hover tbody tr:hover {{
  background-color: #f1f3f5;
}}
</style>
</head>
<body>
<div class="container-fluid">
  <h1>📊 NASDAQ Scan Report</h1>
  <p class="text-muted">Generated on {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
  {html_table}
</div>

<script>
$(document).ready(function() {{
  $('table').DataTable({{
    "pageLength": 25,
    "order": [[1, "desc"]],
    "autoWidth": false
  }});
}});
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 HTML report saved → {output_path}")
