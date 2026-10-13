"""Regenerate docs/repository_map.md from `git ls-files`.

    python tools/update_repository_map.py

Descriptions come from a small table of known files, else from a module's first
docstring line, a YAML file's first comment, or a Markdown file's title.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KNOWN = {
    ".project-root": "Root marker used by `rootutils.setup_root(...)`.",
    ".gitignore": "Ignore rules for generated artefacts, environments, logs, data and vignette outputs.",
    ".pre-commit-config.yaml": "Code-quality hooks (ruff lint + format, file hygiene checks).",
    "LICENSE": "MIT license.",
    "README.md": "Landing page: method summary, install, typical usage, workflow table, citation.",
    "CITATION.cff": "Machine-readable citation (preprint DOI 10.64898/2026.10.08.757722).",
    "CHANGELOG.md": "Release notes.",
    "Makefile": "Convenience targets: `test`, `format`, `clean`, `clean-logs`.",
    "pyproject.toml": "Package metadata, dependencies, extras, console entry points (`grass-mil-*`), pytest/ruff configuration.",
    "environment.yaml": "Conda environment alternative for local development.",
    ".github/PULL_REQUEST_TEMPLATE.md": "Pull-request template.",
    ".github/codecov.yml": "Codecov configuration.",
    ".github/dependabot.yml": "Dependabot update policy.",
    ".github/release-drafter.yml": "Release drafter configuration.",
    ".github/workflows/test.yml": "CI: pytest on Linux and macOS, coverage, install-from-source smoke test.",
    ".github/workflows/code-quality-main.yaml": "CI: pre-commit on every push to main.",
    ".github/workflows/code-quality-pr.yaml": "CI: pre-commit on changed files of a pull request.",
    ".github/workflows/release-drafter.yml": "CI: draft release notes from merged pull requests.",
    ".vscode/settings.json": "Editor settings.",
    "tools/update_repository_map.py": "Regenerates this map.",
    "vignettes/README.md": "Index of the worked examples on the public MIBI-TOF dataset.",
    "vignettes/01_anndata_to_spatial_graphs.ipynb": "Vignette 1: AnnData -> manifest -> cellular graphs.",
    "vignettes/02_pretraining_and_niches.ipynb": "Vignette 2: BGRL pretraining, fine-tuning, instance export, interpretability report.",
    "vignettes/03_nested_cv_niches_and_metaclusters.ipynb": "Vignette 3: nested CV, cell extraction, niches, metaclusters, scores, spatial statistics.",
    "src/grass_mil/configs/local/.gitkeep": "Placeholder for untracked machine-local configs.",
    "tests/conftest.py": "Shared pytest fixtures (Hydra configs on a synthetic datamodule).",
    "tests/helpers/synthetic_cohort.py": "Synthetic cohort on disk (regions, labels, folds) with a planted niche, used by the phase tests.",
    "tests/helpers/synthetic_datamodule.py": "In-memory bag datamodule for fast training tests.",
}

GROUPS = [
    ("Root metadata and tooling", lambda f: "/" not in f),
    ("CI and contribution files", lambda f: f.startswith((".github/", ".vscode/"))),
    ("Maintenance tools", lambda f: f.startswith("tools/")),
    (
        "Package: entry points and shared modules",
        lambda f: f.startswith("src/grass_mil/") and f.count("/") == 2,
    ),
    ("Package: data pipeline", lambda f: f.startswith("src/grass_mil/data/")),
    ("Package: models", lambda f: f.startswith("src/grass_mil/models/")),
    ("Package: inference", lambda f: f.startswith("src/grass_mil/inference/")),
    ("Package: evaluation statistics", lambda f: f.startswith("src/grass_mil/evaluation/")),
    (
        "Package: interpretability (niches, metaclusters, report)",
        lambda f: f.startswith("src/grass_mil/interpretability/"),
    ),
    (
        "Package: classical spatial statistics",
        lambda f: f.startswith("src/grass_mil/spatial_stats/"),
    ),
    ("Package: utilities", lambda f: f.startswith("src/grass_mil/utils/")),
    ("Hydra configs", lambda f: f.startswith("src/grass_mil/configs/")),
    ("Documentation", lambda f: f.startswith("docs/")),
    ("Vignettes", lambda f: f.startswith("vignettes/")),
    ("Tests", lambda f: f.startswith("tests/")),
]


def describe(path: str) -> str:
    if path in KNOWN:
        return KNOWN[path]
    p = ROOT / path
    if p.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(p.read_text()))
        except Exception:
            doc = None
        if doc:
            first = doc.strip().splitlines()[0].strip()
            return first if first.endswith((".", ":")) else first + "."
        if p.name == "__init__.py":
            return "Package marker / re-exports."
        if path.startswith("tests/"):
            return (
                "Tests for "
                + re.sub(r"^tests?[_/]|test_|\.py$", "", p.stem).replace("_", " ")
                + "."
            )
        return "Module."
    if p.suffix in {".yaml", ".yml"}:
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") and not line.startswith("# @package"):
                text = line.lstrip("# ").rstrip()
                return text if text.endswith((".", ":")) else text + "."
            if line and not line.startswith("#"):
                break
        return "Hydra config: `" + p.stem + "`."
    if p.suffix == ".md":
        return p.read_text().strip().splitlines()[0].lstrip("# ").strip() + "."
    if p.suffix == ".ipynb":
        return "Notebook."
    return "Tracked file."


def main() -> None:
    files = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=ROOT
    ).stdout.split()
    files = [f for f in files if (ROOT / f).exists()]
    out = [
        "# Repository Map",
        "",
        "Every tracked file (`git ls-files`) with a one-line purpose. Regenerate with",
        "`python tools/update_repository_map.py` after adding files.",
        "",
    ]
    seen: set[str] = set()
    for title, pred in GROUPS:
        members = [f for f in files if f not in seen and pred(f)]
        if not members:
            continue
        out += [f"## {title}", ""]
        for f in sorted(members):
            seen.add(f)
            out.append(f"- `{f}`: {describe(f)}")
        out.append("")
    rest = [f for f in files if f not in seen]
    if rest:
        out += ["## Other", ""] + [f"- `{f}`: {describe(f)}" for f in sorted(rest)] + [""]
    out += [
        "## Generated at runtime (untracked)",
        "",
        "- `logs/`: Hydra run directories (checkpoints, `metrics_per_epoch.jsonl`, predictions, caches, reports).",
        "- `data/`: raw manifests and processed graphs (`processed_index.json`, `metadata.json`, `graphs/`).",
        "- `vignettes/vignette_work/`: data, graphs and runs produced by the vignettes.",
        "",
    ]
    (ROOT / "docs/repository_map.md").write_text("\n".join(out))
    print(len(files), "files mapped")


if __name__ == "__main__":
    main()
