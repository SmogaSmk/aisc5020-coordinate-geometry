"""Run the Module 2 Adam coordinate-dependence diagnostic.

The replay experiment uses a fixed canonical GD gradient history so that a
rotation changes only Adam's coordinatewise normalization. It is a mechanism
diagnostic, not a tuned optimizer comparison or a convergence bound.
"""

'''
1. Finally select which angles and checkpoints
2. GD history and learning rate and steps
3. Different epsilon? conditional number?or initial scale
4. final charts and reports considered take in
5. connect the 3rd part of the experiment
'''
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform

import numpy as np

from .adam import adam_step, replay_defect
from .contracts import OptimizerState, RunConfig
from .gd_momentum import gd_step
from .problems import QuadraticProblem, matched_pair, rotation_2d


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_gradient_history(
    problem: QuadraticProblem,
    initial: np.ndarray,
    config: RunConfig,

) -> np.ndarray:
    """
    Args:
    - QuadraticProblem provides eigens (n, ), rotations(n, n) and Hessian (n, n) 
    - initial: The initial statement
    - config: Adam initial parameters and running time limitations
    """
    """
    motivation: 提供一组固定且旋转前后严格对应的梯度
    - Return gradients from a canonical GD trajectory before each update.
    - gradients: 
    """

    if config.algorithm != "gd":
        raise ValueError("canonical gradient history requires a GD configuration")
    state = OptimizerState.initial(initial)
    gradients = []
    for _ in range(config.max_steps):
        gradient = problem.gradient(state.position)
        # The position means the local parameter w_t
        gradients.append(gradient.copy())
        gd_step(state, gradient, config)
        # GD here is for updating the position 
        if not np.all(np.isfinite(state.position)):
            raise RuntimeError("canonical GD history became nonfinite; revise the diagnostic configuration")
    return np.asarray(gradients)


def adam_position_history(
    problem: QuadraticProblem,
    initial: np.ndarray,
    config: RunConfig,
) -> np.ndarray:
    # motivation :观察 Adam 实际运行时，旋转前后的轨迹会偏离多少。
    '''
    参数初始化和上一个GD标准旋转初始化时一致
    '''
    """Return every Adam position, including the initial position."""
    if config.algorithm != "adam":
        raise ValueError("Adam position history requires an Adam configuration")
    state = OptimizerState.initial(initial)
    positions = [state.position.copy()]
    for _ in range(config.max_steps):
        gradient = problem.gradient(state.position)
        adam_step(state, gradient, config)
        if not all(
            np.all(np.isfinite(value))
            for value in (state.position, state.first_moment, state.second_moment)
        ):
            raise RuntimeError("Adam diagnostic trajectory became nonfinite; revise the configuration")
        positions.append(state.position.copy())
    return np.asarray(positions)


def diagnostic_records(
    settings: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Compute replay defects and ordinary-Adam correspondence errors."""
    if settings.get("study_status") != "module2_adam_diagnostic":
        raise ValueError("expected study_status='module2_adam_diagnostic'")

    eigenvalues = np.asarray(settings["eigenvalues"], dtype=np.float64)
    if eigenvalues.shape != (2,) or not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0):
        raise ValueError("eigenvalues must contain two finite positive values")
    angles = np.asarray(settings["angles_degrees"], dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("angles_degrees must be a nonempty finite vector")

    max_steps = int(settings["max_steps"])
    checkpoints = [int(value) for value in settings["checkpoints"]]
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoints must be nonempty and unique")
    if any(step < 1 or step > max_steps for step in checkpoints):
        raise ValueError("checkpoints must lie between 1 and max_steps")

    gd_config = RunConfig("gd", float(settings["gd_learning_rate"]), max_steps)
    adam_config = RunConfig(
        "adam",
        float(settings["adam_learning_rate"]),
        max_steps,
        beta1=float(settings["beta1"]),
        beta2=float(settings["beta2"]),
        epsilon=float(settings["epsilon"]),
    )
    rng = np.random.default_rng(int(settings["seed"]))
    canonical, _, initial, _ = matched_pair(
        eigenvalues,
        np.eye(2),
        rng,
        initial_loss=float(settings["initial_loss"]),
    )
    # Because passing a unit matrix I, so there's no need to receive all the values
    gradients = canonical_gradient_history(canonical, initial, gd_config)
    canonical_adam_positions = adam_position_history(canonical, initial, adam_config)
    position_scale = max(1.0, float(np.linalg.norm(initial)))

    rows: list[dict[str, object]] = []
    angle_summaries = []
    for angle in angles:
        rotation = rotation_2d(np.deg2rad(angle))
        rotated = QuadraticProblem(eigenvalues, rotation)
        rotated_initial = rotation @ initial
        defect = replay_defect(gradients, rotation, adam_config)
        rotated_positions = adam_position_history(rotated, rotated_initial, adam_config)
        expected_positions = canonical_adam_positions @ rotation.T
        position_errors = np.linalg.norm(rotated_positions - expected_positions, axis=1) / position_scale

        for index in range(max_steps):
            rows.append(
                {
                    "angle_degrees": float(angle),
                    "step": index + 1,
                    "gradient_norm": float(np.linalg.norm(gradients[index])),
                    "replay_numerator": float(defect["numerator"][index]),
                    "replay_denominator": float(defect["denominator"][index]),
                    "replay_relative_defect": float(defect["relative_defect"][index]),
                    "adam_normalized_position_error": float(position_errors[index + 1]),
                    "is_checkpoint": index + 1 in checkpoints,
                }
            )
        angle_summaries.append(
            {
                "angle_degrees": float(angle),
                "maximum_replay_relative_defect": float(np.max(defect["relative_defect"])),
                "final_replay_relative_defect": float(defect["relative_defect"][-1]),
                "maximum_adam_normalized_position_error": float(np.max(position_errors)),
                "final_adam_normalized_position_error": float(position_errors[-1]),
            }
        )

    problem_record = {
        "eigenvalues": eigenvalues.tolist(),
        "canonical_initial": initial.tolist(),
        "initial_loss": canonical.loss(initial),
        "gradient_history_source": "canonical GD, gradient recorded before each update",
        "position_error_normalization": "max(1, norm(canonical_initial))",
    }
    summary = {
        "study_status": settings["study_status"],
        "protocol_version": settings["protocol_version"],
        "protocol_frozen": False,
        "interpretation": (
            "Mechanism diagnostic only: replay isolates Adam normalization from optimizer "
            "trajectory divergence and is not a convergence bound."
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gd_config": asdict(gd_config),
        "adam_config": asdict(adam_config),
        "checkpoints": checkpoints,
        "angles": angle_summaries,
    }
    return rows, problem_record, summary


def plot_diagnostics(
    rows: list[dict[str, object]],
    checkpoints: list[int],
    output: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Install optional plots dependencies: python -m pip install -e '.[plots]'"
        ) from exc

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for step in checkpoints:
        selected = [row for row in rows if row["step"] == step]
        angles = [row["angle_degrees"] for row in selected]
        axes[0].plot(
            angles,
            [row["replay_relative_defect"] for row in selected],
            marker="o",
            label=f"step {step}",
        )
        axes[1].plot(
            angles,
            [row["adam_normalized_position_error"] for row in selected],
            marker="o",
            label=f"step {step}",
        )
    axes[0].set(
        xlabel="Rotation angle (degrees)",
        ylabel="Replay relative defect",
        title="Fixed canonical GD gradient history",
    )
    axes[1].set(
        xlabel="Rotation angle (degrees)",
        ylabel="Normalized position error",
        title="Ordinary Adam trajectory correspondence",
    )
    for axis in axes:
        axis.set_yscale("symlog", linthresh=1e-14)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "defect_versus_angle.png", dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adam_diagnostics.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/adam_diagnostics"))
    parser.add_argument("--plot", action="store_true", help="Create a figure; requires optional plots dependencies")
    args = parser.parse_args(argv)

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    rows, problem_record, summary = diagnostic_records(settings)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "diagnostics.csv", rows)
    write_json(args.output / "run_config.json", settings)
    write_json(args.output / "problem.json", problem_record)
    write_json(args.output / "summary.json", summary)
    if args.plot:
        plot_diagnostics(rows, [int(value) for value in settings["checkpoints"]], args.output)

    log_lines = [
        f"UTC output timestamp: {datetime.now(timezone.utc).isoformat()}",
        "Module 2 mechanism diagnostic; no tuning, convergence bound or optimizer ranking.",
        f"Config: {args.config}",
        f"Python: {platform.python_version()}; NumPy: {np.__version__}",
        f"Rows: {len(rows)}; angles: {len(summary['angles'])}",
    ]
    (args.output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Adam diagnostics completed: {len(rows)} rows; artifacts in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
