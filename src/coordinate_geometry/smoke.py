"""Integration smoke run, not a tuned comparison or a main-study result."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import platform
from pathlib import Path

import numpy as np

from .contracts import RunConfig
from .problems import matched_pair, rotation_2d
from .runner import run


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--plot", action="store_true", help="Create a figure; requires optional plots dependencies")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings.get("study_status") != "infrastructure_smoke_only":
        parser.error("This entrypoint only accepts infrastructure_smoke_only configurations")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(settings["seed"])
    q = rotation_2d(np.deg2rad(settings["angle_degrees"]))
    canonical, rotated, z0, w0 = matched_pair(np.asarray(settings["eigenvalues"]), q, rng)
    common = {key: settings[key] for key in ("max_steps", "tolerance", "record_every")}
    summary_rows = []
    trace_rows = []
    complete_results = []
    for candidate in settings["algorithms"]:
        config = RunConfig(**common, **candidate)
        for orientation, problem, initial in (("diagonal", canonical, z0), ("rotated", rotated, w0)):
            result = run(problem, initial, config)
            run_id = f"smoke__{orientation}__{config.algorithm}"
            row = {
                "run_id": run_id, "algorithm": config.algorithm, "orientation": orientation,
                "learning_rate": config.learning_rate, "seed": settings["seed"],
                "dimension": problem.dimension, "condition_number": problem.condition_number,
                "status": result.status, "gradient_calls": result.gradient_calls,
                "initial_loss": result.initial_loss, "final_loss": result.final_loss,
                "relative_final_loss": result.final_loss / result.initial_loss,
                "first_attainment": result.first_attainment,
                "threshold_retained": result.threshold_retained,
                "elapsed_seconds": result.metadata["elapsed_seconds"],
            }
            summary_rows.append(row)
            trace_rows.extend({"run_id": run_id, **asdict(entry)} for entry in result.trace)
            complete_results.append({**row, "config": asdict(config),
                                     "final_position": result.final_position.tolist(),
                                     "metadata": result.metadata})
    with (args.output / "run_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output / "traces.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)
    write_json(args.output / "run_config.json", settings)
    write_json(args.output / "problem.json", {
        "eigenvalues": canonical.eigenvalues.tolist(), "rotation": q.tolist(),
        "canonical_initial": z0.tolist(), "rotated_initial": w0.tolist(),
        "normalization": "shared initial loss = 1", "basis_source": "generator oracle",
    })
    write_json(args.output / "summary.json", {
        "study_status": "infrastructure_smoke_only", "protocol_frozen": False,
        "python": platform.python_version(), "numpy": np.__version__,
        "runs": complete_results,
    })
    log_lines = [f"UTC start/output timestamp: {datetime.now(timezone.utc).isoformat()}",
                 "Infrastructure smoke only; no tuning, inferential statistics or performance claims.",
                 f"Config: {args.config}", f"Python: {platform.python_version()}; NumPy: {np.__version__}"]
    log_lines.extend(f"{row['run_id']}: {row['status']}; calls={row['gradient_calls']}; "
                     f"relative_loss={row['relative_final_loss']:.8g}" for row in summary_rows)
    (args.output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("Install optional plots dependencies: python -m pip install -e '.[plots]'") from exc
        figure, axis = plt.subplots(figsize=(8, 5))
        for row in summary_rows:
            values = [item for item in trace_rows if item["run_id"] == row["run_id"]]
            axis.semilogy([v["step"] for v in values],
                          [max(v["relative_loss"], 1e-16) if v["relative_loss"] is not None else np.nan for v in values],
                          label=f"{row['algorithm']} ({row['orientation']})",
                          linestyle="--" if row["orientation"] == "rotated" else "-")
        axis.set(xlabel="Gradient evaluations", ylabel="Relative objective gap",
                 title="Infrastructure smoke check (illustrative rates, no tuning)")
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(args.output / "smoke.png", dpi=160)
        plt.close(figure)
    print(f"Smoke completed: {len(summary_rows)} runs; artifacts in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
