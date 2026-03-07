---
name: python-release-workflow
description: Use when releasing nilspodlib from this repo with uv+poe+GitHub Actions+GitHub Releases+PyPI trusted publishing; supports patch/minor/major and requires CI before creating a release.
---

# Python Release Workflow

## Scope
- Repository: `mad-lab-fau/NilsPodLib`
- Tooling: `uv`, `poe`, `gh`
- Release types: `patch`, `minor`, `major`
- Package layout: single package `nilspodlib`
- Docs hosting: Read the Docs, no manual `gh-pages` publish in this repo

## Inputs
- `bump`: `patch` | `minor` | `major`
- `base_ref`: `main`
- Release notes source: latest top section in `CHANGELOG.md`

## Semver Quick Rule
- `patch`: fixes/docs/tooling/compat, no breaking API
- `minor`: backward-compatible features
- `major`: breaking behavior/API

## Flow
1. **Preflight**
   - `git status -sb` must be clean or intentionally scoped
   - `git log -5 --oneline`

2. **Local verification**
   - `uv sync --group dev`
   - `uv run poe ci_check`
   - `uv run pytest`
   - `uv build`

3. **Release prep**
   - Finalize release notes in `CHANGELOG.md`
   - Bump version: `uv run poe version --bump=<patch|minor|major>`
   - Verify touched files:
     - `pyproject.toml`
     - `src/nilspodlib/__init__.py`
     - `uv.lock`
     - `CHANGELOG.md`

4. **Commit strategy**
   - Commit code/docs/tooling fixes first
   - Commit release prep separately: version + changelog + lockfile

5. **Push + CI gate**
   - `git push`
   - `REL_SHA=$(git rev-parse HEAD)`
   - `gh run list --workflow "Test and Lint" --commit "$REL_SHA" --limit 5`
   - `gh run watch <run-id>`
   - Required: workflow conclusion is `success` for the release SHA

6. **Create GitHub release**
   - Tag format: `vX.Y.Z`
   - `gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes "<release notes from changelog>"`

7. **Post-release checks**
   - Watch the `Upload Python Package` workflow triggered by the release
   - Confirm the PyPI publish job succeeded
   - Confirm the new version is visible on PyPI

## Hard Gates
- Never create a GitHub release before CI is green for the release SHA.
- Never claim the PyPI release is done without command evidence from GitHub Actions or PyPI.
- Never skip local verification before the version bump.

## Fast Command Set
```bash
uv sync --group dev
uv run poe ci_check
uv run pytest
uv build
uv run poe version --bump=patch
git push
REL_SHA=$(git rev-parse HEAD)
gh run list --workflow "Test and Lint" --commit "$REL_SHA" --limit 5
gh run watch <run-id>
gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes "<changelog section>"
```
