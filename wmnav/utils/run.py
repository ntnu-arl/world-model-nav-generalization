"""Run directories, git provenance and the metric sink shared by every entry point.

Every run records the exact code it ran (commit, ``git diff`` against it, and the
list of untracked files) next to its checkpoints. The SSL runs of the IROS26 paper
recorded this only in W&B; when the W&B files were lost so was the code state.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(args, cwd):
    try:
        return subprocess.run(
            ["git"] + args, cwd=cwd, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_state(repo_dir: Path = REPO_ROOT) -> Dict[str, str]:
    """Commit, uncommitted diff and untracked files of the repository."""
    return {
        "commit": _git(["rev-parse", "HEAD"], repo_dir).strip(),
        "diff": _git(["diff", "HEAD"], repo_dir),
        "untracked": _git(["ls-files", "--others", "--exclude-standard"], repo_dir),
    }


def save_git_state(run_dir: Path, repo_dir: Path = REPO_ROOT) -> Dict[str, str]:
    """Write :func:`git_state` into ``run_dir`` (``git_commit.txt``, ``git.diff``,
    ``git_untracked.txt``) and return it.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = git_state(repo_dir)
    (run_dir / "git_commit.txt").write_text(state["commit"] + "\n")
    (run_dir / "git.diff").write_text(state["diff"])
    (run_dir / "git_untracked.txt").write_text(state["untracked"])
    return state


class RunLogger:
    """Writes metrics to ``<run_dir>/metrics.jsonl`` and, optionally, to W&B.

    ``mode`` is the W&B mode: ``online``, ``offline`` (sync later with
    ``wandb sync``) or ``disabled``. The JSONL file is always written, so a run is
    analysable without W&B.
    """

    def __init__(self, run_dir, project: str, name: str, config: Optional[dict] = None,
                 mode: str = "offline"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.run_dir / "metrics.jsonl", "a")  # noqa: SIM115 - open for the run, closed in close()
        self._wandb = None
        if mode != "disabled":
            import wandb

            self._wandb = wandb.init(
                project=project, name=name, config=config or {}, mode=mode,
                dir=str(self.run_dir), reinit=True,
            )

    def log(self, data: Dict[str, float], step: int) -> None:
        """Append ``data`` at ``step`` to ``metrics.jsonl`` (and W&B when enabled)."""
        if not data:
            return
        record = {"step": int(step)}
        record.update({k: float(v) for k, v in data.items()})
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(data, step=int(step))

    def summary(self, data: Dict[str, float]) -> None:
        """Final numbers of a run, also written to ``summary.json``."""
        path = self.run_dir / "summary.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(data)
        path.write_text(json.dumps(current, indent=2))
        if self._wandb is not None:
            for k, v in data.items():
                self._wandb.summary[k] = v

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Close the metrics file and finish the W&B run."""
        self._file.close()
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None
