"""The proxy-vs-true gauge must report Q and the realised return IN THE SAME UNITS
when the critic is split across two discount rates.

WHY THIS TEST EXISTS. A dual-gamma run (v77) reported a Q0-MC "gap" pinned at +28 to
+30 with CQL(alpha=5), the Cal-QL clamp, LayerNorm and dual-gamma all verified active,
and the obvious reading -- "none of them calibrated anything" -- was about to go into a
conclusion. The gauge was comparing two quantities that are not each other's target:

  * Q0 was the min over the WHOLE ensemble, while `CQLSAC.train` has the actor ascend
    min(shaping pair) + min(terminal pair). A min over all four heads is neither stream's
    value and neither the actor's objective.
  * MC was sum gamma^t (r_shaping + r_terminal) at the single `model.gamma`, while the
    split critic represents sum g_s^t r_shaping + sum g_t^t r_terminal.

Both sides were wrong, in different directions, so the number could not be read at all.
The numbers below are chosen so the two conventions disagree -- the old code reports
q=5.0, mc=10.9 on this episode, the correct one q=12.0, mc=12.2608 -- which is what makes
this a test rather than a restatement.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunarsim.rl.monitoring import ProxyVsTrueCallback

G_S, G_T = 0.9, 0.99          # deliberately far apart, and far from 1
HEAD_VALUES = [5.0, 6.0, 7.0, 8.0]   # shaping pair -> min 5, terminal pair -> min 7
R_TOTAL = [1.0, 2.0, 10.0]
R_TERM = [0.0, 0.0, 8.0]


class _StubCritic:
    def __call__(self, s0, a0):
        n = s0.shape[0]
        return [torch.full((n, 1), v) for v in HEAD_VALUES]


class _StubLogger:
    name_to_value: dict = {}


class _StubModel:
    gamma = G_S
    dual_gamma = (G_S, G_T)
    device = "cpu"
    num_timesteps = 100
    critic = _StubCritic()
    logger = _StubLogger()
    _last_obs = np.zeros((1, 20), dtype=np.float32)


def _run_episode(csv_path):
    cb = ProxyVsTrueCallback(log_every=10**9, csv_path=str(csv_path), verbose=0)
    cb.model = _StubModel()
    cb._on_training_start()
    for t, (r, rt) in enumerate(zip(R_TOTAL, R_TERM)):
        done = t == len(R_TOTAL) - 1
        cb.locals = {
            "rewards": np.array([r], dtype=np.float32),
            "dones": np.array([done]),
            "infos": [{"reward_terminal": rt, "reward_shaping": r - rt,
                       "landed_safely": done}],
            "actions": np.zeros((1, 4), dtype=np.float32),
        }
        cb._on_step()
    cb._flush()
    import csv as _csv
    with open(csv_path) as fh:
        return list(_csv.DictReader(fh))[-1]


def test_mc_uses_each_streams_own_discount(tmp_path):
    row = _run_episode(tmp_path / "g.csv")
    expected = sum(G_S ** t * (R_TOTAL[t] - R_TERM[t]) for t in range(3)) \
        + sum(G_T ** t * R_TERM[t] for t in range(3))
    assert expected == pytest.approx(12.2608, abs=1e-4)
    # The single-gamma convention would give 10.9 on this episode. A tolerance looser
    # than that difference would let the defect through, so it is pinned tight.
    assert float(row["mc_s0"]) == pytest.approx(expected, abs=1e-4)


def test_q0_is_the_scalar_the_actor_ascends(tmp_path):
    row = _run_episode(tmp_path / "q.csv")
    half = len(HEAD_VALUES) // 2
    expected = min(HEAD_VALUES[:half]) + min(HEAD_VALUES[half:])
    assert expected == pytest.approx(12.0)
    # min over the whole ensemble -- what the gauge used to report -- is 5.0.
    assert float(row["q_s0"]) == pytest.approx(expected, abs=1e-5)
    assert float(row["q_s0"]) != pytest.approx(min(HEAD_VALUES), abs=1e-3)


def test_single_gamma_path_is_unchanged(tmp_path):
    """No dual_gamma attribute -> the old single-gamma convention, exactly."""
    class _Plain(_StubModel):
        dual_gamma = None
    cb = ProxyVsTrueCallback(log_every=10**9, csv_path=str(tmp_path / "p.csv"), verbose=0)
    cb.model = _Plain()
    cb._on_training_start()
    for t, r in enumerate(R_TOTAL):
        cb.locals = {
            "rewards": np.array([r], dtype=np.float32),
            "dones": np.array([t == len(R_TOTAL) - 1]),
            "infos": [{"landed_safely": False}],
            "actions": np.zeros((1, 4), dtype=np.float32),
        }
        cb._on_step()
    cb._flush()
    import csv as _csv
    with open(tmp_path / "p.csv") as fh:
        row = list(_csv.DictReader(fh))[-1]
    assert float(row["mc_s0"]) == pytest.approx(sum(G_S ** t * R_TOTAL[t] for t in range(3)))
    assert float(row["q_s0"]) == pytest.approx(min(HEAD_VALUES))
