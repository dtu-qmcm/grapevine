# Anonymisation Plan for Double-Blind Peer Review

## Overview
This plan outlines the steps to anonymise the current `grapevine` repository branch for double-blind peer review. The current branch is the intended anonymised version. The `.git` folder will not be submitted for review.

Wherever identifying information is found, it should be replaced with "ANONYMISED"

## Files Requiring Anonymisation

### 1. `pyproject.toml`
**Current identifying information:**
- Line 6: `{name = "Teddy Groves", email = "tedgro@dtu.dk"}`

**Actions:**
- Replace author name with: `{name = "ANONYMISED", email = "ANONYMISED@example.com"}`

### 2. `README.md`
**Current identifying information:**
- Line 2: `[![Tests](https://github.com/dtu-qmcm/grapevine/actions/workflows/run_tests.yml/badge.svg)](https://github.com/dtu-qmcm/grapevine/actions/workflows/run_tests.yml)`

**Actions:**
- Remove or replace GitHub badges with generic text
- Ensure no author names, emails, or institutional affiliations remain

### 3. `LICENSE`
**Current identifying information:**
- Line 3: `Copyright (c) 2025 Quantitative Modelling of Cell Metabolism`

**Actions:**
- Replace copyright holder with: `Copyright (c) 2025 ANONYMISED`

## Step-by-Step Implementation

### Phase 1: Verify Current Branch
1. Confirm current branch:
   ```bash
   git branch --show-current
   ```

### Phase 2: Apply Anonymisation Changes
1. Edit `pyproject.toml`:
   - Replace author information

2. Edit `README.md`:
   - Remove/replace GitHub badges and repository references

3. Edit `LICENSE`:
   - Replace copyright holder

### Phase 3: Verification
1. Check for remaining identifying information (should return no results):
     ```bash
     grep -r "Teddy\|teddy\|tedgro\|Quantitative Modelling" --include="*.py" --include="*.md" --include="*.txt" --include="*.toml" . --exclude-dir=.git
     ```

2. Verify ANONYMISED text was applied:
    ```bash
    grep -r "ANONYMISED" pyproject.toml README.md LICENSE
    ```

3. Verify author in pyproject.toml
4. Verify README.md badges
5. Verify LICENSE copyright holder

### Phase 4: Create Archive for Review

**Option 1 (recommended): Use git archive**
- Automatically excludes .git folder and any untracked files
- Ensure all required files are committed/tracked:
    ```bash
    git status
    git add <any untracked files>
    git commit -m "Add files for anonymised submission"
    ```
- Create archive from current branch:
    ```bash
    git archive anonymised | gzip > anonymised-grapevine.tar.gz
    ```

**Option 2: Use tar with exclusions**
- Manually specify directories to exclude:
    ```bash
    tar -czf anonymised-grapevine.tar.gz --exclude='.git' --exclude='.venv' --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='*.pyc' .
    ```

## Files NOT Requiring Changes

- Source code in `src/grapevine/` (no author comments found)
- Test files in `tests/`
- Benchmark files in `benchmarks/`
- GitHub Actions workflow files (no personal info in content)
- Package metadata (name, version, description are fine)

## Post-Review Restoration (Optional)

After peer review, if needed:
1. Merge anonymised branch back to main or discard
2. Restore original author information in files
3. Tag the review version for reference

## Checklist

- [ ] Verify current branch is correct
- [ ] Anonymise pyproject.toml
- [ ] Anonymise README.md
- [ ] Anonymise LICENSE
- [ ] Verify no identifying information remains
- [ ] Test that code still functions correctly
- [ ] Create archive excluding .git folder for submission

## Notes

- Current branch is called "anonymised" and is the intended anonymised version
- The .git folder will not be included in the review submission, so git history is not a concern
- Git archive is preferred for creating the submission as it automatically excludes .git and untracked files
- Ensure all required files are tracked/committed before using git archive
- The enzax dependency from dtu-qmcm GitHub can remain in pyproject.toml (reviewers will not draw conclusions from this)
- Ensure the anonymised version passes all tests before submission
- Archive should exclude: .git, .venv, .pytest_cache, .ruff_cache, and any other temporary/cache directories
