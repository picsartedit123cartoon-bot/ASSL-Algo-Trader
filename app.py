from flask import Flask, render_template, jsonify
from pathlib import Path
import csv

app = Flask(__name__)

BASE = Path(__file__).parent
REPORTS = BASE / "reports"


def read_csv(filename):
    file = REPORTS / filename

    if not file.exists():
        return []

    with file.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/data")
def data():
    analysis = read_csv("latest_analysis.csv")
    trades = read_csv("paper_trades.csv")

    return jsonify({
        "paper_mode": True,
        "analysis": analysis[-1] if analysis else {},
        "trades": trades[-20:]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
