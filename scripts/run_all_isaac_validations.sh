#!/usr/bin/env bash
# Runs every Isaac-Sim-backed validation script in sequence against a real
# container and reports a final pass/fail summary. One-command regression
# check for everything that needs a live Isaac Sim process (structural
# authoring, physics behavior, sensor cross-validation, lighting math).
#
# Usage: ./scripts/run_all_isaac_validations.sh
set -uo pipefail  # not -e: we want to run every script and collect results

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
chmod -R a+rwX out/ 2>/dev/null || true
mkdir -p out

SCRIPTS=(
  "scripts/isaac_smoke_test.py:structural (heightfield/materials/lighting/rocks/sensors authoring)"
  "scripts/isaac_test_sun_rotation.py:sun light direction math (pure transform check, no render)"
  "scripts/isaac_test_rock_collision.py:rock collision (real ball-drop physics)"
  "scripts/isaac_test_dust.py:dust particle trajectories (real ballistic authoring)"
  "scripts/isaac_validation_suite.py:LiDAR ground truth vs. PhysX + physics step speed + camera"
)

declare -a RESULTS
for entry in "${SCRIPTS[@]}"; do
  script="${entry%%:*}"
  desc="${entry#*:}"
  echo ""
  echo "=== $script ($desc) ==="
  if ./scripts/run_isaac_smoke_test.sh "$script"; then
    RESULTS+=("PASS  $script")
  else
    RESULTS+=("FAIL  $script")
  fi
  chmod -R a+rwX out/ 2>/dev/null || true
done

echo ""
echo "================ SUMMARY ================"
n_fail=0
for r in "${RESULTS[@]}"; do
  echo "$r"
  [[ "$r" == FAIL* ]] && n_fail=$((n_fail + 1))
done
echo "==========================================="
if [[ $n_fail -eq 0 ]]; then
  echo "All ${#RESULTS[@]} validation scripts passed."
else
  echo "$n_fail of ${#RESULTS[@]} validation scripts FAILED."
fi
exit $n_fail
