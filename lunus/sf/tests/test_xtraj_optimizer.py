"""
xtraj's frame-selection optimizer: the whole sweep, and the one cctbx seam.

test_selection_matches_a_naive_reference is the test that matters. It runs the
entire greedy selection twice -- once through sweep_candidates() and
correlator(), once through a naive loop that shares none of their shortcuts --
and asserts the same frames come out in the same order. That single comparison
covers the expanded diffuse expression, the blocking, complex64 storage, the
reduced AnisoOperator and the segment-weighted correlation together, which is
why the unit tests for those pieces individually are gone.

The mapping from calculated to experimental reflections used to be tested
separately here, when it lived in its own function. It is inlined in the
trajectory loop now, and the coverage moved rather than vanished: feeding
test_xtraj_smoke.py a SHUFFLED SUBSET as ID_file makes the two match columns
different permutations, so a swapped or reordered mapping fails there. That
was established by mutation, not assumed -- see the commit that made the
change.

Everything here needs cctbx, and importing xtraj needs mdtraj, h5py and gemmi
besides; the module skips rather than errors where they are missing.
"""
import numpy as np
import pytest

from lunus.sf.aniso import AnisoOperator, subtract_isotropic


@pytest.fixture(scope="module")
def xtraj():
    return pytest.importorskip(
        "lunus.command_line.xtraj",
        reason="xtraj needs cctbx, mdtraj, h5py and gemmi")


# --------------------------------------------------------------------------
# the end-to-end test
# --------------------------------------------------------------------------

def _trajectory(n_frames=60, n_refl=3000, diffuse_to_bragg=1e-3, seed=5):
    """
    A synthetic trajectory with a coherent Bragg component, so that
    n*sum|F|^2 - |sum F|^2 is the cancellation it is in practice, plus a
    partition with Laue-like multiplicities and a diffuse-looking target.
    """
    rng = np.random.default_rng(seed)
    d_star = rng.uniform(0.02, 0.55, n_refl)
    shell_thickness = 0.05

    multiplicities = []
    while sum(multiplicities) < n_refl:
        multiplicities.append(int(rng.integers(1, 17)))
    multiplicities[-1] -= sum(multiplicities) - n_refl
    segments = np.repeat(np.arange(len(multiplicities)), multiplicities)
    rng.shuffle(segments)
    _, segments = np.unique(segments, return_inverse=True)
    segments = segments.ravel().astype(np.int64)

    bragg = (rng.normal(size=n_refl) + 1j * rng.normal(size=n_refl)) * 100.0
    spread = np.abs(bragg).mean() * np.sqrt(diffuse_to_bragg)
    fcalc = bragg[None, :] + (rng.normal(size=(n_frames, n_refl))
                              + 1j * rng.normal(size=(n_frames, n_refl))) * spread

    experiment = 500.0 * np.exp(-8.0 * d_star) + rng.normal(0.0, 20.0, n_refl)
    return d_star, shell_thickness, segments, fcalc, experiment


def _naive_aniso(diffuse, d_star, shell_thickness, segments):
    """subtract_isotropic() at full length, then a plain mean per segment."""
    out = subtract_isotropic(diffuse, d_star, shell_thickness)
    means = np.array([out[segments == g].mean() for g in range(segments.max() + 1)])
    return means[segments]


def test_selection_matches_a_naive_reference(xtraj):
    """
    The naive path evaluates one candidate at a time with the unexpanded
    n*(I_tot - I_x) - |F_tot - F_x|^2 in complex128, removes the isotropic
    component at full length, averages segments with a plain mean and
    correlates with np.corrcoef. The fast path stores F at complex64,
    evaluates the expanded form in blocks, reduces to segment means and
    carries the multiplicities as weights.

    Identical removals, in identical order, is the claim.
    """
    d_star, thickness, segments, fcalc64, experiment = _trajectory()
    n_frames, n_refl = fcalc64.shape

    def naive():
        tot_f = fcalc64.sum(axis=0)
        tot_i = (np.abs(fcalc64) ** 2).sum(axis=0)
        weights = np.ones(n_frames)
        n, removed = n_frames, []
        diffuse = (n * tot_i - (tot_f * tot_f.conjugate()).real)
        best = np.corrcoef(experiment,
                           _naive_aniso(diffuse, d_star, thickness, segments))[0, 1]
        while True:
            n -= 1
            correlations = np.zeros(n_frames)
            for x in np.nonzero(weights)[0]:
                sig_f = tot_f - fcalc64[x]
                sig_i = tot_i - np.abs(fcalc64[x]) ** 2
                diffuse = n * sig_i - (sig_f * sig_f.conjugate()).real
                correlations[x] = np.corrcoef(
                    experiment,
                    _naive_aniso(diffuse, d_star, thickness, segments))[0, 1]
            if correlations.max() <= best:
                return removed
            best = correlations.max()
            k = int(correlations.argmax())
            tot_f -= fcalc64[k]
            tot_i -= np.abs(fcalc64[k]) ** 2
            weights[k] = 0
            removed.append(k)

    def fast(block_size):
        stored = fcalc64.astype(np.complex64)          # as the optimizer stores it
        operator = AnisoOperator(d_star, thickness, segment_index=segments,
                                 reduced=True)
        correlate = xtraj.correlator(experiment, operator)
        tot_f = np.sum(stored, axis=0, dtype=np.complex128)
        tot_i = np.zeros(n_refl)
        for x in range(n_frames):
            tot_i += np.abs(stored[x].astype(np.complex128)) ** 2
        weights = np.ones(n_frames)
        n, removed = n_frames, []
        diffuse = n * tot_i - (tot_f * tot_f.conjugate()).real
        best = correlate(diffuse[None, :])[0]
        while True:
            n -= 1
            correlations = xtraj.sweep_candidates(
                stored, tot_f, tot_i, weights, n, correlate, block_size)
            if correlations.max() <= best:
                return removed
            best = correlations.max()
            k = int(correlations.argmax())
            tot_f -= stored[k]
            tot_i -= np.abs(stored[k].astype(np.complex128)) ** 2
            weights[k] = 0
            removed.append(k)

    expected = naive()
    assert len(expected) > 5, "fixture converged too early to be a test"
    for block_size in (1, 7, n_frames):
        assert fast(block_size) == expected, "block_size=%d" % block_size


def test_sweep_skips_deselected_frames(xtraj):
    """Frames whose weight is zero are not evaluated and come back zero."""
    d_star, thickness, segments, fcalc64, experiment = _trajectory()
    n_frames, n_refl = fcalc64.shape
    stored = fcalc64.astype(np.complex64)
    correlate = xtraj.correlator(
        experiment, AnisoOperator(d_star, thickness, segment_index=segments,
                                  reduced=True))
    weights = np.ones(n_frames)
    weights[::3] = 0

    correlations = xtraj.sweep_candidates(
        stored, np.sum(stored, axis=0, dtype=np.complex128),
        (np.abs(fcalc64) ** 2).sum(axis=0), weights, n_frames - 1, correlate, 7)

    assert np.all(correlations[::3] == 0.0)
    assert np.all(correlations[weights != 0] != 0.0)


def test_correlator_without_an_operator_is_plain_pearson(xtraj):
    """The corr_aniso=False path is np.corrcoef, batched."""
    rng = np.random.default_rng(0)
    reference = rng.normal(100.0, 30.0, 500)
    rows = rng.normal(size=(6, 500)) * 50.0

    np.testing.assert_allclose(
        xtraj.correlator(reference)(rows),
        [np.corrcoef(reference, row)[0, 1] for row in rows],
        rtol=0, atol=1e-12)
