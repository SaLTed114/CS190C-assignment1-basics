from __future__ import annotations

from pathlib import Path
import json
import time

from tqdm import tqdm

class JsonlLogger:
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.messages_path = self.out_dir / "messages.jsonl"
        self.summaries_path = self.out_dir / "summaries.jsonl"

        self.start_time = time.time()

    def log_metrics(self, metrics: dict) -> None:
        row = dict(metrics)
        row["wall_time_sec"] = time.time() - self.start_time
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_message(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
        tqdm.write(line)
        with self.messages_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_summary(self, summary: dict) -> None:
        with self.summaries_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")