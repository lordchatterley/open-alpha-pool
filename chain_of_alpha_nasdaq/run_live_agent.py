import os
import time
import schedule
import sys
from datetime import datetime
from chain_of_alpha_nasdaq.agents.live_agent import LiveAlphaAgent

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_message(message: str):
    """Append timestamped message to today’s log file and print to console."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{ts} UTC] {message}"
    print(log_entry)
    log_file = os.path.join(LOG_DIR, f"live_agent_{datetime.utcnow().date()}.log")
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")

def run_job():
    log_message("🚀 Running scheduled LiveAlphaAgent job...")
    try:
        agent = LiveAlphaAgent(
            tickers=["AAPL", "MSFT"],  # later: full NASDAQ list
            db_path="alphas.db",
        )
        agent.run_once()
        log_message("✅ Run complete.")
    except Exception as e:
        log_message(f"❌ ERROR during run: {e}")

def setup_schedule():
    schedule_time = "21:00"  # ~NASDAQ close (UTC)
    log_message(f"🕒 Scheduling LiveAlphaAgent to run daily at {schedule_time} UTC")
    schedule.every().day.at(schedule_time).do(run_job)

    # Immediate test run
    log_message("🔧 Running initial test now...")
    run_job()

    log_message("⏰ Scheduler active. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    try:
        setup_schedule()
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped manually.")
        sys.exit(0)
