from __future__ import annotations

from pathlib import Path

from app.training.live import parse_results_csv, read_log_tail


def test_parse_results_csv_reads_epochs_and_latest(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,time,train/box_loss,metrics/mAP50(B),val/box_loss\n"
        "1,8.1,1.8,0.1,1.2\n"
        "2,16.2,1.5,0.3,1.0\n",
        encoding="utf-8",
    )
    epochs, series, latest = parse_results_csv(csv_path)
    assert epochs == [1, 2]
    assert series["train/box_loss"] == [1.8, 1.5]
    assert latest["metrics/mAP50(B)"] == 0.3
    assert latest["train/box_loss"] == 1.5


def test_read_log_tail_returns_last_lines(tmp_path: Path):
    log_path = tmp_path / "logs.txt"
    log_path.write_text("\n".join(f"line-{i}" for i in range(100)) + "\n", encoding="utf-8")
    lines = read_log_tail(log_path, max_lines=5)
    assert lines == [f"line-{i}" for i in range(95, 100)]
