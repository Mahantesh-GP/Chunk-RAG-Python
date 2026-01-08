# DevOps Task Plan and Estimates

This plan outlines actionable DevOps tasks with manager-friendly estimates. It’s organized by phases, includes task descriptions, and rolls up section totals for easy tracking.

- Base scope total: 66 hours
- Optional Azure integration scope: +12 hours
- Potential grand total: 78 hours

---

## Phase 1 — Repository & Environment (8h)
- [ ] Standardize Python environment and dependency locking — 3h
  - Description: Ensure `.venv` usage is consistent; lock dependencies (e.g., `pip-tools` or pinned `requirements.txt`). Verify `python -m rag_eval.cli` works in clean environments.
- [ ] Add pre-commit hooks (format/lint) — 2h
  - Description: Configure `pre-commit` with `black`, `isort`, `ruff/flake8`. Enforce on commit and in CI.
- [ ] Add `.env.example` and dotenv check — 1h
  - Description: Provide sample env config and verify `python-dotenv` loads in local runs without breaking when missing.
- [ ] Task runner for common workflows — 2h
  - Description: Add `Makefile` (WSL) and `invoke`/PowerShell script for Windows to run `format`, `lint`, `test`, `run`.

Section total: 8h

---

## Phase 2 — CI/CD Foundation (14h)
- [ ] GitHub Actions: Lint + Test workflow (matrix) — 4h
  - Description: CI on PRs/main for Windows/Ubuntu (or Python 3.10–3.12 matrix). Cache dependencies.
- [ ] Coverage reporting & status check — 2h
  - Description: Generate coverage, upload artifact, and set minimum threshold (e.g., 80%) as a check.
- [ ] Build distributable artifact on tags — 3h
  - Description: Build wheel/sdist and upload as workflow artifacts to validate packaging.
- [ ] PR templates and required checks — 2h
  - Description: Add `.github/PULL_REQUEST_TEMPLATE.md`; document and enable required checks for merge.
- [ ] Branch protection rules doc — 3h
  - Description: Document desired protection rules (reviews, checks, linear history) for admin enablement.

Section total: 14h

---

## Phase 3 — Testing & Quality Gates (10h)
- [ ] Improve unit test coverage to ~85% — 6h
  - Description: Expand tests for `evaluator.py` (relevancy/faithfulness), retrieval, and chunking edge cases.
- [ ] Golden-file test for `evaluation_results.json` — 2h
  - Description: Stable input → stable output test to guard against regressions.
- [ ] Integration test for PDF extraction — 2h
  - Description: Add small sample PDF; verify `load_documents_from_dir` extracts text cleanly.

Section total: 10h

---

## Phase 4 — Packaging & Versioning (6h)
- [ ] Semantic versioning automation — 3h
  - Description: Adopt `commitizen` or `bump2version` to consistently version releases.
- [ ] Changelog automation — 3h
  - Description: Auto-generate `CHANGELOG.md` from conventional commits or release notes.

Section total: 6h

---

## Phase 5 — Security & Compliance (8h)
- [ ] Dependency updates and policy — 1h
  - Description: Enable Dependabot (or similar) for Python dependencies with update cadence.
- [ ] SAST with Bandit — 2h
  - Description: Add Bandit to pre-commit and CI to detect common Python security issues.
- [ ] Secrets scanning — 3h
  - Description: Add `detect-secrets` pre-commit and ensure GitHub secret scanning is enabled.
- [ ] License/compliance scan — 2h
  - Description: Use `pip-licenses` (or equivalent) to generate license inventory in CI.

Section total: 8h

---

## Phase 6 — Containerization & Dev Environments (8h)
- [ ] Dockerfile + .dockerignore — 3h
  - Description: Multi-stage image to run CLI (`rag_eval.cli`) and tests consistently.
- [ ] Dev container (`.devcontainer/`) — 2h
  - Description: VS Code devcontainer for consistent local dev with extensions and tasks.
- [ ] CI build & publish image (internal) — 3h
  - Description: Build image in CI and publish to GitHub Container Registry as a snapshot.

Section total: 8h

---

## Phase 7 — Observability & Artifacts (6h)
- [ ] Structured logging and levels — 2h
  - Description: Standardize logging format (ISO timestamps, levels) and configurable verbosity.
- [ ] Persist evaluation artifacts in CI — 2h
  - Description: Upload `evaluation_results.json` and logs as CI artifacts for each run.
- [ ] HTML results export — 2h
  - Description: Enhance `show_results.py` (or a new script) to render an HTML report from JSON.

Section total: 6h

---

## Phase 8 — Documentation & Governance (6h)
- [ ] README refresh with badges — 2h
  - Description: Add CI/coverage badges, quickstart, and troubleshooting links.
- [ ] Developer guide & Ops runbook — 3h
  - Description: Steps for local dev, CI/CD flows, releases, and incident handling basics.
- [ ] CONTRIBUTING + CODEOWNERS — 1h
  - Description: Contribution guidelines and ownership for reviews/areas.

Section total: 6h

---

## Phase 9 — Optional: Azure Integration Roadmap (+12h)
- [ ] Feature-flagged embeddings via Azure OpenAI — 5h
  - Description: Add real embeddings provider behind an interface; keep MockEmbedder as default.
- [ ] Azure Cognitive Search adapter — 5h
  - Description: Pluggable retriever for ACS; environment-based configuration and fallback.
- [ ] IaC & pipeline placeholders — 2h
  - Description: Draft infra/pipeline docs (Bicep/Terraform repo structure, secrets strategy) for later execution.

Section total: 12h (optional)

---

## Rollup Summary
- Phase 1: 8h
- Phase 2: 14h
- Phase 3: 10h
- Phase 4: 6h
- Phase 5: 8h
- Phase 6: 8h
- Phase 7: 6h
- Phase 8: 6h
- Optional Phase 9: +12h

Base scope total: 66h
Optional Azure scope: +12h
Potential grand total: 78h

---

## Notes & Usage
- Checkboxes can be used to track progress by phase.
- Estimates assume 1 engineer; parallel work reduces calendar time.
- Convert items into GitHub Issues with labels: `phase:<n>`, `type:devops`, `priority`.
- Add owners and target dates as appropriate for your team.
