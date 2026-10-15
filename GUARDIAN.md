# GUARDIAN

Automated security / bug-hygiene ledger for this repo (branch `claude/guardian`).

## Checks

| Area | Tool / rule | Command |
|---|---|---|
| Deps | pip-audit (resolves unpinned `requirements.txt`) | `pip-audit -r requirements.txt` |
| Deps | osv-scanner | `osv-scanner -r .` (API blocked by sandbox proxy this run) |
| Code | bandit | `bandit -r symphony -ll` and `bandit -r symphony` |
| Code | grep rules | `grep -rnE "eval\(\|exec\(\|pickle\|torch.load\|yaml.load\|shell=True\|verify=False\|0\.0\.0\.0\|subprocess\|os.system" symphony notebooks setup.py` |
| Secrets | gitleaks (history + tree) | `gitleaks detect --source . --redact -v`; `gitleaks dir . --redact -v` |
| Secrets | detect-secrets | `detect-secrets scan --all-files .` |
| CI | GitHub Actions review | none present (`.github/` absent) |
| Docker | Dockerfile review | none present |
| Tests | pytest (CPU, tensorflow-cpu) | `python -m pytest tests -q` |
| Lint | ruff | `ruff check .` |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | high | bug: `train.py` passed `vocab_size` into `create_sequences`' `num_samples` slot, silently truncating training to 128 sequences | symphony/sourcing/sourcing.py | fixed | 2026-10-15 |
| G-002 | medium | bug: no `__init__.py`, so `find_packages()` installed zero packages | setup.py, symphony/ | fixed | 2026-10-15 |
| G-003 | medium | bug: `MyModel` used a divergent `mse_with_positive_pressure` that gives zero loss for under-prediction | symphony/model/model.py | fixed | 2026-10-15 |
| G-004 | low | bug: bare `except:` caught `SystemExit` (`--help`) and required undeclared `ipdb` | symphony/train2.py | fixed | 2026-10-15 |
| G-005 | low | bug: `midi_to_notes` raised opaque `IndexError` on MIDI without notes | symphony/preproccesing/preprocessing.py | fixed | 2026-10-15 |
| G-006 | low | bug: `download_maestro_dataset(data_dir)` passed a path as the `colab` flag; `print(dir)` printed the builtin | symphony/sourcing/sourcing.py | fixed | 2026-10-15 |
| G-007 | low | info exposure: tracked log files / `.DS_Store` / egg-info leaking local directory layout | repo root | fixed (untracked + .gitignore) | 2026-10-15 |
| G-008 | low | missing dep `ipython` (imported by manager), duplicate `pretty_midi` | requirements.txt | fixed | 2026-10-15 |
| G-009 | low | notebook imported non-existent `symphony.manager.AudiManager` | notebooks/test.ipynb | fixed | 2026-10-15 |
| G-010 | low | supply chain: all requirements unpinned; dataset zip downloaded without `file_hash` integrity check | requirements.txt, sourcing*.py | open | 2026-10-15 |
| G-011 | low | hard-coded absolute developer paths as defaults | train.py, train2.py, preprocessing.py, sourcing2.py | open | 2026-10-15 |
| G-012 | low | ~3 MB of TF checkpoints tracked despite `.gitignore` | training_checkpoints/ | open | 2026-10-15 |

No secrets found by gitleaks (history + tree) or detect-secrets. bandit: only B101 (`assert`, low).

## Metrics

| metric | before | after |
|---|---|---|
| open vulns crit/high/med/low (deps) | 0/0/0/0 | 0/0/0/0 |
| open findings crit/high/med/low (all) | 0/1/2/9 | 0/0/0/3 |
| failing tests | n/a (no tests) | 0 |
| test count | 0 | 6 |
| ruff errors (`ruff check .`) | 30 | 16 |
| bandit (all severities) | 1 low | 1 low |

## Backlog

- Pin dependency versions (or add a lockfile) and pass `file_hash` to `tf.keras.utils.get_file` (G-010).
- Replace absolute `/Users/...` default paths with CLI args / env vars (G-011).
- `git rm --cached training_checkpoints/` and `output.mid` (G-012) once the owner confirms they are not needed.
- Deduplicate `train.py`/`train2.py` and `sourcing.py`/`sourcing2.py`; `train.py` runs top-level training on import.
- Remaining ruff findings (unused imports, useless expressions in train.py).

## Learnings

- Positional-argument drift between `sourcing.py` and `sourcing2.py` `create_sequences` signatures caused silent data truncation; keep one canonical helper.
- TensorFlow-cpu installs fine in the sandbox (~1 min), enough for tiny CPU regression tests.

## Ideas

1. Add a minimal GitHub Actions workflow (SHA-pinned, `permissions: contents: read`) running pytest + ruff on tensorflow-cpu.
2. Add a 1-epoch smoke-training test on a synthetic MIDI to catch model/loss wiring regressions.
3. Consolidate the duplicate training scripts into one argparse CLI.

## Log

- Run 1 — 2026-10-15 — Initial scan (pip-audit, osv-scanner, bandit, gitleaks, detect-secrets, ruff); fixed 9 findings (1 high, 2 medium, 6 low) with 6 regression tests; untracked logs/egg-info/.DS_Store; 3 low items left in backlog.
