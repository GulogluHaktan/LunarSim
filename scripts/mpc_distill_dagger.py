"""Distil the terrain-aware MPPI-MPC into a controller-free MLP (DAgger) on the analytic backend.

Student sees exactly what the ES policy sees: 20 env obs + 7 hazard-radar features (27-d).
Labels = MPC action at the replan states; iteration 0 flies the expert, later iterations fly a
student/expert mixture (beta decays), so the student is trained on its own state distribution.
Saves an MLPPolicy-compatible npz (theta, sizes, mu, sd) so evo_isaac_eval / ES fine-tune can load it.
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.evo.mpc import MPPIMPC, MPCConfig
from lunarsim.evo.policies import MLPPolicy
from lunarsim.evo.hazfeat import N_HAZ, haz_features
from lunarsim.core.terrain.hazard import attach_hazard
from scripts.mpc_analytic_eval import make_env

p = argparse.ArgumentParser()
p.add_argument("--stage", default="orbit_descent")
p.add_argument("--mpc-params", required=True)
p.add_argument("--hidden", default="64,64")
p.add_argument("--iters", type=int, default=4)
p.add_argument("--batches", type=int, default=3)
p.add_argument("--n", type=int, default=16)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--bias", type=float, default=-0.05)
p.add_argument("--seed0", type=int, default=300000)
p.add_argument("--out", required=True)
a = p.parse_args()
hidden = tuple(int(x) for x in a.hidden.split(","))
OBS = 20 + N_HAZ
pol = MLPPolicy(OBS, 4, hidden, a.bias)
lp = np.clip(np.load(a.mpc_params), -3.0, 4.0)
cfg = MPCConfig(K=512, M=20, hold=12, exec_steps=4, roll_sub=2)

def student_act(theta, mu, sd, o):
    return np.stack([pol.act(theta, (oi - mu) / sd) for oi in o])

def collect(seeds, theta, mu, sd, beta):
    from lunarsim.rl.reward import RewardWeights
    n = len(seeds); envs = []; obs = []
    for s in seeds:
        e, st = make_env(a.stage, int(s)); o, _ = e.reset(seed=int(s)); envs.append(e); obs.append(np.asarray(o, float))
    mpc = MPPIMPC(st.params, RewardWeights(), cfg, lp); mpc.reset(n)
    tl = [attach_hazard(e.tile) for e in envs]; mpc.set_hazard([t.haz_dist for t in tl], tl[0].size_m, tl[0].res_m, [t.height for t in tl])
    done = np.zeros(n, bool); landed = np.zeros(n, bool); X, Y = [], []
    cap = int(st.params.max_episode_s / st.params.dt_s) + 5; stepi = 0
    use_exp = np.random.default_rng(stepi + int(seeds[0])).random(n) < beta
    while not done.all() and stepi < cap:
        S = {k: np.zeros(n) for k in ("alt", "vx", "vy", "vz", "tilt_x", "tilt_y", "yaw", "wx", "wy", "wz", "fuel_kg")}
        for i, e in enumerate(envs):
            s_ = e.state
            for k in S: S[k][i] = (s_["z"] - e._ground_z(s_["x"], s_["y"])) if k == "alt" else s_[k]
        S["t_left"] = np.array([st.params.max_episode_s - e._t for e in envs])
        S["px"] = np.array([e.state["x"] for e in envs]); S["py"] = np.array([e.state["y"] for e in envs])
        A = mpc.act(S)
        feats = []
        for i, e in enumerate(envs):
            s_ = e.state
            f = np.concatenate([obs[i], haz_features(s_["x"], s_["y"], s_["vx"], s_["vy"], tl[i].haz_dist, tl[i].res_m)])
            feats.append(f)
            if not done[i]:
                y = A[i].copy(); y[0] -= a.bias; X.append(f); Y.append(y)
        feats = np.array(feats)
        for _ in range(cfg.exec_steps):
            for i, e in enumerate(envs):
                if done[i]: continue
                act = A[i] if (theta is None or use_exp[i]) else student_act(theta, mu, sd, [np.concatenate(
                    [obs[i], haz_features(e.state["x"], e.state["y"], e.state["vx"], e.state["vy"], tl[i].haz_dist, tl[i].res_m)])])[0]
                o, r, te, tr, info = e.step(act); obs[i] = np.asarray(o, float)
                if te or tr: done[i] = True; landed[i] = bool(info["landed_safely"])
            stepi += 1
    return np.array(X), np.array(Y), landed

def fit(X, Y, theta, mu, sd):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev); Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    layers = []; sizes = pol.sizes
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2: layers.append(torch.nn.Tanh())
    net = torch.nn.Sequential(*layers).to(dev)
    if theta is not None:   # warm start from previous student
        k = 0
        for m in net:
            if isinstance(m, torch.nn.Linear):
                ai, bi = m.in_features, m.out_features
                m.weight.data = torch.tensor(theta[k:k + ai * bi].reshape(ai, bi).T, dtype=torch.float32, device=dev); k += ai * bi
                m.bias.data = torch.tensor(theta[k:k + bi], dtype=torch.float32, device=dev); k += bi
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(Xn)
    for ep in range(a.epochs):
        perm = torch.randperm(n, device=dev)
        for j in range(0, n, 512):
            ix = perm[j:j + 512]; loss = torch.nn.functional.mse_loss(net(Xn[ix]), Yt[ix])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): final = torch.nn.functional.mse_loss(net(Xn), Yt).item()
    out = []
    for m in net:
        if isinstance(m, torch.nn.Linear):
            out += [m.weight.detach().cpu().numpy().T.ravel(), m.bias.detach().cpu().numpy()]
    return np.concatenate(out).astype(np.float64), final

os.makedirs(a.out, exist_ok=True)
Xall, Yall = [], []; theta = None; mu = np.zeros(OBS); sd = np.ones(OBS); t0 = time.time(); seed = a.seed0
for it in range(a.iters):
    beta = 1.0 if it == 0 else max(0.1, 0.5 - 0.1 * it)
    lands = []
    for b in range(a.batches):
        seeds = seed + np.arange(a.n); seed += a.n
        X, Y, L = collect(seeds, theta, mu, sd, beta); Xall.append(X); Yall.append(Y); lands += list(L)
    XA, YA = np.concatenate(Xall), np.concatenate(Yall)
    if it == 0:
        mu = XA.mean(0); sd = XA.std(0) + 1e-2
    theta, mse = fit(XA, YA, theta, mu, sd)
    pol.save(os.path.join(a.out, "student.npz"), theta, mu=mu, sd=sd)
    print(f"iter {it} beta={beta:.2f} rollout_land={np.mean(lands):.2f} data={len(XA)} mse={mse:.4f} t={time.time()-t0:.0f}s", flush=True)
