#!/bin/bash
# PPO blend-method experiment (laptop). Seed-major order so partial results stay balanced.
#   arms: smoothstep, wendland, gaussian_calibrated (3 seeds); fine = no-blend reference (seed 1 only)
#   per run: train 15,360 decisions -> policy-only validation, 192 held-out episodes (16 waves x 12 envs, wave seeds 90000..)
# Usage: ROOT=/path/to/LunarSim DEM=/path/to/lroc POOLS=/path/to/train_pools VALPOOLS=/path/to/val_pools OUT=/path/to/out ./run_ppo_queue.sh
set -u
ROOT=${ROOT:-/home/haktan/Projects/LunarSim}; DEM=${DEM:-/home/haktan/lunarsim_data/lroc}
POOLS=${POOLS:-$ROOT/out/paper/pools_train}; VALPOOLS=${VALPOOLS:-$ROOT/out}; OUT=${OUT:-$ROOT/out/paper/ppo}
PY=${PY:-/home/haktan/isaac-env/bin/python}
cd $ROOT; export PYTHONPATH=$ROOT:${EXTRA_PP:-}; mkdir -p $OUT
valpool() { echo $VALPOOLS/$1/pool_val.pkl; }
dem() { echo $DEM/NAC_DTM_$(echo $1 | sed 's/a/APOLLO/').TIF; }
trainpool() { if [ "$2" = fine ]; then echo $VALPOOLS/$1/pool_train.pkl; else echo $POOLS/${1}_$2.pkl; fi; }
# 0) controller baseline on the same held-out episodes (once per site)
for site in a16 a11; do
  [ -f $OUT/controller_$site.done ] && continue
  LUNARSIM_LOLA_IMG=$(dem $site) LUNARSIM_TILE_POOL=$(valpool $site) $PY scripts/validate_guided_gain_isaac.py \
    --stage orbit_descent --gain-set attitude --hold-steps 20 --n-envs 12 --waves 16 --wave-seed0 90000 \
    --out-json $OUT/controller_$site.json > $OUT/controller_$site.log 2>&1 && touch $OUT/controller_$site.done
done
for seed in 1 2 3; do for site in a16 a11; do for method in smoothstep wendland gaussian_calibrated fine; do
  [ "$method" = fine ] && [ $seed != 1 ] && continue
  tag=${method}_${site}_s$seed; [ -f $OUT/$tag.done ] && continue
  s=$(date +%s)
  export LUNARSIM_LOLA_IMG=$(dem $site)
  LUNARSIM_TILE_POOL=$(trainpool $site $method) $PY scripts/train_guided_gain_isaac.py --stage orbit_descent --gain-set attitude \
    --hold-steps 20 --gamma 0.995 --gae-lambda 0.99 --n-steps 64 --batch-size 192 --steps 15360 --n-envs 12 --seed $seed \
    --eval-episodes 24 --eval-every 5120 --out $OUT/$tag > $OUT/$tag.train.log 2>&1
  t1=$(date +%s)
  LUNARSIM_TILE_POOL=$(valpool $site) $PY scripts/validate_guided_gain_isaac.py $OUT/$tag/best_on_selection_block \
    --stage orbit_descent --gain-set attitude --hold-steps 20 --n-envs 12 --waves 16 --wave-seed0 90000 --policy-only \
    --out-json $OUT/$tag.val.json > $OUT/$tag.val.log 2>&1
  echo "$tag train_s=$((t1-s)) val_s=$(( $(date +%s)-t1 ))" >> $OUT/summary.txt; touch $OUT/$tag.done
done; done; done
echo ALLDONE >> $OUT/summary.txt
