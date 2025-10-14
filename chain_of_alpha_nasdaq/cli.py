import typer
from rich.table import Table
from rich.console import Console

from chain_of_alpha_nasdaq import agent, alpha_db

app = typer.Typer()
console = Console()


@app.command()
def run(
    tickers: str,
    start: str,
    end: str,
    num_new_factors: int = typer.Option(3, help="Number of new factors to generate"),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI model for factor generation"),
    export_csv: str = typer.Option(None, "--export-csv", help="Export factors + metrics to CSV file"),
):
    """
    Run the Chain-of-Alpha pipeline for given tickers and date range.
    """
    typer.echo(f"🚀 Running Chain-of-Alpha with tickers={tickers}, {start} → {end}")

    caa = agent.ChainOfAlphaAgent(tickers=tickers, model=model)
    results = caa.run_pipeline(start=start, end=end, num_new_factors=num_new_factors)

    # Normalize tickers: support comma-separated or multiple args
    if len(tickers) == 1 and "," in tickers[0]:
        tickers = [t.strip() for t in tickers[0].split(",")]

    if results:
        typer.echo("✅ Run completed")

        if export_csv:
            db = alpha_db.AlphaDatabase()
            db.export_to_csv(export_csv)
            typer.echo(f"📂 Exported results to {export_csv}")

@app.command("list-runs")
def list_runs():
    """
    List all recorded runs with their run_id, timestamp, and factor count.
    """
    db = alpha_db.AlphaDB()
    runs = db.list_runs()
    db.close()

    if not runs:
        typer.echo("📭 No runs found in database.")
        raise typer.Exit()

    typer.echo("📊 Recorded Runs:")
    for run_id, timestamp, num_factors in runs:
        typer.echo(f" - {run_id} | {timestamp} | {num_factors} factors")



@app.command("show-factors")
def show_factors(
    limit: int = 20,
    db_path: str = "alpha.db",
    export: str = typer.Option(None, "--export", help="Export results to CSV file"),
):
    """Display stored factors and their latest metrics (optionally export to CSV)."""
    import csv

    db = alpha_db.AlphaDB(db_path=db_path)
    rows = db.query_latest_metrics(limit=limit)
    db.close()

    if not rows:
        console.print("⚠️ No factors found in DB")
        return

    if export:
        with open(export, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["name", "formula", "IC", "RankIC", "Sharpe", "Turnover", "run_id"],
            )
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"📤 Exported {len(rows)} rows to {export}")
        return

    # Pretty table view
    table = Table(title="Alpha Pool (latest metrics)", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Formula", style="magenta")
    table.add_column("IC", justify="right")
    table.add_column("RankIC", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Turnover", justify="right")
    table.add_column("Run ID", style="dim")

    for row in rows:
        table.add_row(
            row["name"],
            (row["formula"] or "")[:40] + ("..." if row["formula"] and len(row["formula"]) > 40 else ""),
            f"{row['IC']:.3f}" if row["IC"] is not None else "—",
            f"{row['RankIC']:.3f}" if row["RankIC"] is not None else "—",
            f"{row['Sharpe']:.3f}" if row["Sharpe"] is not None else "—",
            f"{row['Turnover']:.3f}" if row["Turnover"] is not None else "—",
            row["run_id"][:8] if row["run_id"] else "—",
        )

    console.print(table)