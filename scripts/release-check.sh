#!/usr/bin/env bash
# Release gate for sgl-node. Every check that must pass before a binary ships,
# in one command, so the gate lives in the repo instead of in someone's memory.
#
#   ./scripts/release-check.sh          # full gate
#   ./scripts/release-check.sh --fast   # skip cargo-audit (no network)
#
# Exit code is non-zero if ANY check fails. Runs every check before reporting,
# so one failure doesn't hide the others.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1

FAILED=()
PASSED=()
SKIPPED=()

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
run() {
  local name="$1"; shift
  bold "▶ $name"
  if "$@"; then
    PASSED+=("$name")
  else
    FAILED+=("$name")
  fi
  echo
}

bold "sgl-node release check"
echo "repo: $(pwd)"
echo

# Feature note: never `--all-features` here. `metal` is macOS-only and `inprocess`
# builds llama.cpp via cmake. Default features cover the billing/setup/orchestrator
# logic this gate exists to protect. Build the engine features explicitly when a
# change touches them: cargo test --features inprocess

# ── Formatting ───────────────────────────────────────────────────────────────
# Advisory until the one-time `cargo fmt --all` lands (the repo predates any
# formatting pass, ~208 diffs). Promote to a hard gate in that same commit.
bold "▶ cargo fmt --check (advisory)"
if cargo fmt --all -- --check >/dev/null 2>&1; then
  PASSED+=("cargo fmt")
  echo "  formatting clean — promote this to a hard gate (see CLAUDE.md)"
else
  SKIPPED+=("cargo fmt (advisory: repo not yet formatted)")
  echo "  unformatted — advisory only, not blocking this release"
fi
echo

# ── Lints ────────────────────────────────────────────────────────────────────
# Advisory while 9 pre-existing cosmetic lints are outstanding (see
# docs/audit/FINDINGS.md). Once cleared, make this `run` with `-- -D warnings`
# so a new lint blocks the release — this is a money path.
bold "▶ cargo clippy (advisory)"
if cargo clippy --all-targets -- -D warnings >/dev/null 2>&1; then
  PASSED+=("cargo clippy")
  echo "  clean — promote this to a hard gate (see CLAUDE.md)"
else
  SKIPPED+=("cargo clippy (advisory: known lint debt)")
  echo "  lints outstanding — advisory only, not blocking this release"
fi
echo

# ── Tests ────────────────────────────────────────────────────────────────────
run "cargo test" cargo test

# ── Known-vulnerable dependencies ────────────────────────────────────────────
if [[ $FAST -eq 1 ]]; then
  SKIPPED+=("cargo audit (--fast)")
elif command -v cargo-audit >/dev/null 2>&1; then
  run "cargo audit" cargo audit
else
  SKIPPED+=("cargo audit (not installed — 'cargo install cargo-audit')")
fi

# ── Tree hygiene: never ship from a dirty or unpushed tree ───────────────────
bold "▶ tree hygiene"
DIRTY="$(git status --porcelain)"
if [[ -n "$DIRTY" ]]; then
  echo "  uncommitted changes present:"
  echo "$DIRTY" | sed 's/^/    /'
  FAILED+=("tree hygiene (uncommitted changes)")
else
  BRANCH="$(git branch --show-current)"
  if git rev-parse --verify --quiet "origin/$BRANCH" >/dev/null; then
    AHEAD="$(git rev-list --count "origin/$BRANCH..HEAD")"
    if [[ "$AHEAD" != "0" ]]; then
      echo "  $AHEAD commit(s) ahead of origin/$BRANCH — push before releasing"
      FAILED+=("tree hygiene (unpushed commits)")
    else
      echo "  clean and in sync with origin/$BRANCH"
      PASSED+=("tree hygiene")
    fi
  else
    echo "  no upstream for '$BRANCH' — push it before releasing"
    FAILED+=("tree hygiene (no upstream)")
  fi
fi
echo

# ── Summary ──────────────────────────────────────────────────────────────────
bold "── summary ──"
for p in "${PASSED[@]:-}";  do [[ -n "$p" ]] && echo "  PASS  $p"; done
for s in "${SKIPPED[@]:-}"; do [[ -n "$s" ]] && echo "  SKIP  $s"; done
for f in "${FAILED[@]:-}";  do [[ -n "$f" ]] && echo "  FAIL  $f"; done
echo

if [[ ${#FAILED[@]} -gt 0 ]]; then
  bold "NO-GO — ${#FAILED[@]} check(s) failed."
  echo "Record the blocker + fix owner in docs/audit/FINDINGS.md before moving on."
  exit 1
fi

bold "GO — all checks passed."
echo "Reminder: a rebuilt binary needs its sha256 added to the orchestrator"
echo "ALLOWED_NODE_BINARY_HASHES allowlist, or nodes will stop serving."
