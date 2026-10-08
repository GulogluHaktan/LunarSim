cd /home/haktan/Projects/LunarSim
PY=/home/haktan/isaac-env/bin/python

PIPE_PID="${1:?pass the PID to wait for}"
while kill -0 "$PIPE_PID" 2>/dev/null; do sleep 60; done
for _ in $(seq 1 240); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${FREE:-0}" -ge 3000 ] && break
  sleep 60
done

# ===================================================================== #
# CAN PPO LEARN FROM SCRATCH? Two runs, because one would confound two
# different failure modes.
#
# From-scratch SAC was measured at 0.0% on orbit_descent across two 400k runs, and the
# mechanism was identified: thrust-to-weight at the ACTION-BOX CENTRE is 1.0159, so an
# untrained policy's mean action is almost exactly a hover -- it reached the ground in 0
# of 20 episodes with a median minimum altitude of 200.0 m over 60 s. That is an
# EXPLORATION failure, not a credit-assignment one, and PPO's on-policy advantage does
# nothing about it: a policy that never descends never observes a landing, so there is no
# return for any estimator to learn from.
#
# So the two runs separate the two questions:
#   A) orbit_descent  -- does PPO's estimator alone beat the hover attractor? Predicted
#      NO, for the exploration reason above. Worth measuring rather than asserting,
#      because it is the TARGET stage and the prediction is cheap to falsify.
#   B) land_handoff   -- released at vz in [-5,-1], already descending, so exploration
#      does reach the ground. This is the fair test of "can PPO learn a landing from
#      nothing", and the same stage the SAC specialist runs on, so the two are comparable.
#
# Two differences from the warm-started PPO runs, both deliberate:
#   * no --warm-start, so log_std starts at SB3's default 0.0 (std 1.0). That is the right
#     thing for exploration, and the reason v84 needed lr=1e-5 does not apply here: the
#     1/sigma^2 amplification that made one minibatch overshoot by 765x comes from the
#     clone's sigma = 0.031. At sigma = 1.0 the normal 3e-4 is appropriate.
#   * --ent-coef 0.01 rather than 0.003, since there is no clone to protect and
#     exploration is the binding constraint.
#
# AND gamma 0.999, not the 0.99 these runs were first written with. Measured on 480 demo
# episodes, the terminal reward's weight in a gamma=0.99 return is 0.786 at touchdown but
# 0.0054 at a 200 m release -- and the lateral speed that causes 94% of the failures is
# already +5.86 m/s worse in the failing episodes at 100-200 m. So at 0.99 the part of the
# objective that separates a landing from a crash is invisible exactly where the decision
# is made, and a failure would have been uninterpretable:
#
#     gamma    return std   noise at a state   terminal signal   signal/noise
#     0.990       2.27          0.272               0.004            0.01
#     0.999       7.63          0.693               2.260            3.26
#
# The noise grows 2.5x and the signal 565x. SAC could not use 0.998 because a bootstrap
# fixed point amplifies a per-step error by 1/(1-gamma); PPO's value is a regression onto
# the realised return, so that failure mode does not apply.
# ===================================================================== #

for STAGE in land_handoff orbit_descent; do
  echo "########## PPO from scratch: $STAGE ##########"
  $PY scripts/train_sac_isaac.py --algo ppo \
    --only-stage "$STAGE" --steps-per-stage 1000000 --checkpoint-every 250000 \
    --max-rocks-per-env 0 --learning-rate 3e-4 \
    --gamma 0.999 --ppo-gae-lambda 0.99 --ppo-n-steps 512 --ppo-n-epochs 10 \
    --ppo-batch-size 1024 --ppo-clip-range 0.2 --ent-coef 0.01 \
    --ppo-target-kl 0.03 \
    --proxy-log-every 20000 \
    --out-dir "out/ppo_scratch_${STAGE}" \
    > "out/train_ppo_scratch_${STAGE}.log" 2>&1
  echo "=== rc=$? ==="
  grep -E "^\[ppo\]|^=== stage:" "out/train_ppo_scratch_${STAGE}.log" | head -3
  grep -oE "step +[0-9]+.*landed +[0-9.]+%.*" "out/train_ppo_scratch_${STAGE}.log" \
    | sed -E 's/ +/ /g; s/\(worst10%[^)]*\) //; s/lost=0 left=0 //; s/Q~-?0 //' | sed -n '1~3p'

  echo "--- deterministic, last snapshot, both blocks ---"
  CK=$(ls -1 "out/ppo_scratch_${STAGE}/snapshots/"*.zip 2>/dev/null | tail -1)
  if [ -n "$CK" ]; then
    for s0 in 7000 41000; do
      printf 'scratch_%-14s seed0=%-6s ' "$STAGE" "$s0"
      timeout 2400 $PY scripts/diag_stage_landing_rate.py --checkpoint "$CK" \
        --stage "$STAGE" --episodes 96 --seed0 "$s0" --terrain-seed 7 2>/dev/null \
        | grep -oE "[0-9]+/96 landed_safely \([0-9]+%\), [0-9]+ crash, [0-9]+ timeout" \
        || echo "NO OUTPUT"
    done
  else
    echo "(no snapshot -- the run produced nothing)"
  fi
done
echo "########## done ##########"
