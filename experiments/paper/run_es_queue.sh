#!/bin/bash
# ES blend-method experiment (HP desktop, Apollo 11 only). 3 methods x 3 seeds, seed-major.
#   per run: 18 generations (8 antithetic pairs x 4 episodes), selection blocks at generations 9 and 18 -> policy-only validation (192 episodes)
set -u
ROOT=${ROOT:?}; DEM=${DEM:?}; POOLS=${POOLS:?}; VALPOOL=${VALPOOL:?}; OUT=${OUT:?}; PY=${PY:?}; NENVS=${NENVS:-12}
cd $ROOT; export PYTHONPATH=$ROOT:${EXTRA_PP:-} OMNI_KIT_ACCEPT_EULA=YES; mkdir -p $OUT
export LUNARSIM_LOLA_IMG=$DEM/NAC_DTM_APOLLO11.TIF
if [ ! -f $OUT/controller_a11.done ]; then
  LUNARSIM_TILE_POOL=$VALPOOL $PY scripts/validate_guided_gain_isaac.py --stage orbit_descent --gain-set attitude --hold-steps 20 \
    --n-envs 12 --waves 16 --wave-seed0 90000 --out-json $OUT/controller_a11.json > $OUT/controller_a11.log 2>&1 && touch $OUT/controller_a11.done
fi
for seed in 1 2 3; do for method in smoothstep wendland gaussian_calibrated; do
  tag=${method}_a11_s$seed; [ -f $OUT/$tag.done ] && continue
  s=$(date +%s)
  LUNARSIM_TILE_POOL=$POOLS/a11_$method.pkl $PY scripts/evo_guided_aim_train_isaac.py --gens 18 --pairs 8 --eps-per-cand 4 \
    --sel 24 --sel-every 9 --n-envs $NENVS --seed $seed --out $OUT/$tag > $OUT/$tag.train.log 2>&1
  t1=$(date +%s)
  LUNARSIM_AIM_SCALE=15 LUNARSIM_TILE_POOL=$VALPOOL $PY scripts/validate_guided_gain_isaac.py $OUT/$tag/best_on_selection.npz \
    --stage orbit_descent --gain-set attitude --hold-steps 20 --n-envs 12 --waves 16 --wave-seed0 90000 --policy-only \
    --out-json $OUT/$tag.val.json > $OUT/$tag.val.log 2>&1
  echo "$tag train_s=$((t1-s)) val_s=$(( $(date +%s)-t1 ))" >> $OUT/summary.txt; touch $OUT/$tag.done
done; done
echo ALLDONE >> $OUT/summary.txt
