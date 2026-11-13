"""How study_taper.py says an L-BFGS run ended.

torch's LBFGS leaves its loop on the iteration budget, on max_eval (which
defaults to max_iter * 5 // 4 when unset), or -- with both tolerances at
zero, as run_lbfgs sets them -- on a line search that made no progress.
run_lbfgs used to call every run short of its iterations a failed line
search; a run that ended on the evaluation cap is not one.
"""

import io
import os
import sys

import pytest

from conftest import REPO_ROOT

pytest.importorskip("cctbx")
pytest.importorskip("torch")

TOOLS = os.path.join(REPO_ROOT, "lunus", "sf", "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

FIVE_ATOMS = """\
CRYST1   20.000   24.000   28.000  90.00  90.00  90.00 P 1
ATOM      1  N   SER A   1       2.000   3.000   4.000  1.00 20.00           N
ATOM      2  CA  SER A   1       3.400   3.300   4.200  1.00 20.00           C
ATOM      3  C   SER A   1       4.100   2.100   4.900  1.00 20.00           C
ATOM      4  O   SER A   1       3.500   1.100   5.300  1.00 20.00           O
ATOM      5  CB  SER A   1       4.100   3.700   2.900  1.00 20.00           C
END
"""


def displaced_run(tmp_path, steps, **kwargs):
    """run_lbfgs on five atoms displaced by 0.3 A rms, fitting the model's
    own amplitudes. Returns (record, printed line)."""
    import torch
    import study_taper as st

    path = tmp_path / "five.pdb"
    path.write_text(FIVE_ATOMS)
    s = st.load_system(str(path), 3.0, "cpu", torch.float64, io.StringIO())
    k = st.build_kernels(s, 1e-3, st.HARD)
    disp_cart = 0.3 / 3 ** 0.5 * torch.randn(
        5, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    frac_start = st.displaced(s, s["frac0"], disp_cart).detach()
    with torch.no_grad():
        target = st.forward(s, k, s["frac0"]).abs()
    log = io.StringIO()
    record, _ = st.run_lbfgs(s, k, frac_start, disp_cart, target, steps, 0.1,
                             log, **kwargs)
    return record, log.getvalue()


def test_stop_reason_iteration_budget():
    from study_taper import lbfgs_stop_reason
    assert lbfgs_stop_reason(300, 340, 300, 375) == "iteration budget"


def test_stop_reason_evaluation_cap():
    from study_taper import lbfgs_stop_reason
    assert lbfgs_stop_reason(280, 375, 300, 375) == "evaluation cap"
    # a line search may overshoot the cap by its last evaluations
    assert lbfgs_stop_reason(280, 377, 300, 375) == "evaluation cap"


def test_stop_reason_failed_line_search():
    from study_taper import lbfgs_stop_reason
    assert lbfgs_stop_reason(52, 120, 300, 375) == "failed line search"
    # no iteration at all: the gradient was exactly zero at the start
    assert lbfgs_stop_reason(0, 1, 300, 375) == "failed line search"


def test_stop_reason_iteration_budget_is_tested_first():
    """torch checks n_iter == max_iter before the evaluation count, so a run
    that exhausts both on its last iteration ended on the iterations."""
    from study_taper import lbfgs_stop_reason
    assert lbfgs_stop_reason(300, 375, 300, 375) == "iteration budget"
    assert lbfgs_stop_reason(0, 1, 0, 0) == "iteration budget"


def test_default_max_eval_is_torchs():
    """default_max_eval restates torch's rule; this fails if torch changes it."""
    import torch
    from study_taper import default_max_eval
    for steps in (0, 1, 3, 200, 300):
        p = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([p], max_iter=steps)
        assert opt.param_groups[0]["max_eval"] == default_max_eval(steps)


def test_run_lbfgs_default_names_the_evaluation_cap(tmp_path):
    """Two iterations with max_eval unset: torch's cap is 2 * 5 // 4 = 2
    evaluations, the starting one and the first line search's first already
    make 2, so the run ends after one iteration with one to spare. That is
    stopped early, but on the cap, and not a failed line search."""
    record, line = displaced_run(tmp_path, steps=2)
    assert "STOPPED EARLY: evaluation cap" in line, \
        "run that ended on max_eval is not reported as such: " + line.strip()
    assert record["iterations"] == 1 and record["stopped_early"]
    assert record["evaluations"] >= record["max_eval"] == 2
    assert record["stop_reason"] == "evaluation cap"


def test_run_lbfgs_default_cap_and_old_fields(tmp_path):
    """Unset, max_eval is torch's default, stored in the record; the fields
    the record had before are all still there."""
    record, line = displaced_run(tmp_path, steps=4)
    assert record["max_eval"] == 5
    assert set(record) >= {"loss", "rmsd", "final_loss", "final_rmsd",
                           "iterations", "evaluations", "stopped_early",
                           "evals_to_reach", "max_eval", "stop_reason"}
    assert record["stopped_early"] == (record["iterations"] < 4)
    assert record["stop_reason"] in line


def test_run_lbfgs_max_eval_lifts_the_cap(tmp_path):
    """The same run with room to evaluate uses both its iterations."""
    record, line = displaced_run(tmp_path, steps=2, max_eval=1000)
    assert record["iterations"] == 2 and not record["stopped_early"]
    assert record["max_eval"] == 1000
    assert record["stop_reason"] == "iteration budget"
    assert "ended: iteration budget" in line and "STOPPED EARLY" not in line


def test_run_lbfgs_names_a_failed_line_search(tmp_path, monkeypatch):
    """A loss whose value is right and whose gradient points uphill: the
    first line search finds no step, and the run ends with iterations and
    evaluations both to spare."""
    import study_taper as st
    true_loss = st.loss_fn

    def uphill(F, target):
        L = true_loss(F, target)
        return 2 * L.detach() - L

    monkeypatch.setattr(st, "loss_fn", uphill)
    record, line = displaced_run(tmp_path, steps=20)
    assert record["iterations"] < 20 and record["stopped_early"]
    assert record["evaluations"] < record["max_eval"] == 25
    assert record["stop_reason"] == "failed line search"
    assert "STOPPED EARLY: failed line search" in line
