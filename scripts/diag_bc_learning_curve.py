"""Will MORE demonstrations actually help the clone, and by how much?

The imitation residual splits train 0.1188 / held-out 0.1646 on 120 episodes, which says
coverage is the binding limit. It does not say what another 480 episodes would buy. A
learning curve does: fit the same network on a growing number of EPISODES against a
FIXED held-out set, and read the slope. A curve that is still falling at the right-hand
end means collecting more is worth the hour; one that has flattened means it is not, and
the gap is capacity or the teacher's hidden state instead.

Analytic env, because this is about the teacher's input-output relation and 480 Isaac
episodes is exactly the cost we are deciding whether to pay. Absolute residuals differ
between backends; the SLOPE is what the decision turns on.

MEASURED (300 analytic episodes, 254 landed, 512x512, 300 epochs, 50 landed episodes
held out):

    landed episodes   transitions    train    HELD-OUT
                 25         18452   0.1456      0.2484
                 51         37855   0.1282      0.1902   (-23.4%)
                102         74509   0.0842      0.1351   (-29.0%)
                204        148310   0.0707      0.0936   (-30.7%)

Still falling at the right-hand end, and the gain per doubling is not shrinking. The real
clone is fitted on 48 landed episodes -- the second row -- so collecting ~240 should cut
the held-out action residual roughly in half. The train-to-held-out ratio closes with it,
1.71 -> 1.32, which is the signature of a coverage limit rather than a capacity one.

CACHE: the collected rollouts are cached (see CACHE below). Delete that file to recollect
after any change to the controller, the stage or the observation layout -- a stale cache
would answer the question for an older teacher.
"""
import os
import sys

sys.path.insert(0, "/home/haktan/Projects/LunarSim")
import numpy as np
import torch as th
import torch.nn as nn

CACHE = os.environ.get("BC_CURVE_CACHE", "out/analytic_demos_for_curve.npz")
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 300

if not os.path.exists(CACHE):
    from lunarsim.control.zemzev_controller import ZemZevController, ZemZevGains
    from lunarsim.core.terrain.generate import generate_tile
    from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv
    from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config

    STAGE = STAGES_BY_NAME["orbit_descent"]
    tile = generate_tile(terrain_config(STAGE, 7))
    O, A, EP, LAND = [], [], [], []
    for ep in range(N_EP):
        env = AnalyticLanderEnv(tile=tile, params=STAGE.params,
                                reward_fn=lambda e, i: 0.0)
        ctrl = ZemZevController(STAGE.params, ZemZevGains())
        obs, _ = env.reset(seed=7000 + ep)
        ctrl.reset()
        info = {}
        n0 = len(O)
        for _ in range(int(STAGE.params.max_episode_s / STAGE.params.dt_s) + 5):
            a = ctrl.act(env)
            O.append(np.asarray(obs, np.float32))
            A.append(np.asarray(a, np.float32))
            EP.append(ep)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        LAND.extend([bool(info.get("landed_safely"))] * (len(O) - n0))
        if (ep + 1) % 50 == 0:
            print(f"  collected {ep + 1}/{N_EP} episodes", flush=True)
    np.savez_compressed(CACHE, obs=np.stack(O), actions=np.stack(A),
                        ep=np.asarray(EP), landed=np.asarray(LAND))

d = np.load(CACHE)
O, A, EP, LAND = d["obs"], d["actions"], d["ep"], d["landed"]
from lunarsim.rl.obs_norm import normalize_obs  # noqa: E402

X = normalize_obs(O)
n_ep = int(EP.max()) + 1
print(f"{len(X)} transitions, {n_ep} episodes, "
      f"{len(set(EP[LAND].tolist()))} of them landed")

# The real clone is fitted on LANDED episodes only (train_bc_from_demos --landed-only),
# so the curve is measured on the same subset or it would answer a different question.
keep = LAND
X, A, EP = X[keep], A[keep], EP[keep]
landed_eps = np.array(sorted(set(EP.tolist())))
rng = np.random.default_rng(0)
landed_eps = rng.permutation(landed_eps)
n_test = max(4, len(landed_eps) // 5)
test_eps = set(landed_eps[:n_test].tolist())
pool_eps = landed_eps[n_test:]
te = np.array([e in test_eps for e in EP])
print(f"held out {n_test} landed episodes ({te.sum()} transitions); "
      f"pool has {len(pool_eps)}")


def fit(tr_mask, epochs=300, seed=0):
    th.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(),
                        nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, A.shape[1]))
    opt = th.optim.Adam(net.parameters(), lr=1e-3)
    Xt, At = th.as_tensor(X[tr_mask]), th.as_tensor(A[tr_mask])
    Xv, Av = th.as_tensor(X[te]), th.as_tensor(A[te])
    n = len(Xt)
    for _ in range(epochs):
        perm = th.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            loss = ((net(Xt[idx]) - At[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with th.no_grad():
        return (float(th.norm(net(Xt) - At, dim=1).mean()),
                float(th.norm(net(Xv) - Av, dim=1).mean()))


print("\nlearning curve (512x512, 300 epochs, fixed held-out set):")
print(f"  {'landed episodes':>16} {'transitions':>12} {'train':>9} {'HELD-OUT':>10}")
prev = None
for frac in (0.125, 0.25, 0.5, 1.0):
    k = max(2, int(len(pool_eps) * frac))
    use = set(pool_eps[:k].tolist())
    mask = np.array([e in use for e in EP])
    tr_err, te_err = fit(mask)
    delta = "" if prev is None else f"  ({(te_err / prev - 1) * 100:+.1f}%)"
    print(f"  {k:>16} {int(mask.sum()):>12} {tr_err:>9.4f} {te_err:>10.4f}{delta}",
          flush=True)
    prev = te_err
