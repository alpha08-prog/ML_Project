#!/usr/bin/env bash
set -euo pipefail

MAIN_BRANCH="main"

# Try to fetch latest state of main from origin
if ! git fetch origin "$MAIN_BRANCH" --quiet 2>/dev/null; then
  echo "[branch-check] Warning: could not reach origin. Skipping up-to-date check."
  exit 0
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Skip the check when committing directly on main
if [ "$CURRENT_BRANCH" = "$MAIN_BRANCH" ]; then
  echo "[branch-check] On main branch — skipping up-to-date check."
  exit 0
fi

# Fail if origin/main has commits that are not in the current branch
if ! git merge-base --is-ancestor "origin/$MAIN_BRANCH" HEAD; then
  echo ""
  echo "[branch-check] FAIL: '$CURRENT_BRANCH' is out of date with origin/$MAIN_BRANCH."
  echo "               Run one of the following before committing:"
  echo ""
  echo "                 git pull origin $MAIN_BRANCH          # merge"
  echo "                 git rebase origin/$MAIN_BRANCH        # rebase"
  echo ""
  exit 1
fi

echo "[branch-check] OK: '$CURRENT_BRANCH' is up to date with origin/$MAIN_BRANCH."
