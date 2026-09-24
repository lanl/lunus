"""
AnisoOperator: the precomputed, batched form of xtraj.to_aniso().

to_aniso() re-solves a spline and re-derives the Laue merge mapping -- through
cctbx's merge_equivalents and a per-reflection Python loop -- on every call.
Both depend only on the reflection list, so a leave-one-out sweep repeats the
same work thousands of times. AnisoOperator precomputes it.

The segment average is asserted EXACTLY, against a copy of cctbx's
merge_equivalents_real summation order; the operator as a whole to ~1e-9,
since the background becomes S @ shell_means where subtract_isotropic() solves
and evaluates a spline -- the same linear map reassociated.

Batching, the reduced form in use, and the segment-weighted correlation are
covered end to end by test_xtraj_optimizer.py rather than here.
"""
import numpy as np
import pytest

from lunus.sf.aniso import (
    AnisoOperator,
    segment_average,
    segment_counts,
    subtract_isotropic,
)


def _have(name):
    try:
        __import__(name)
    except ImportError:
        return False
    return True


HAVE_CCTBX = _have("cctbx")


def reference_segment_average(data, segment_index):
    """
    cctbx's merge-then-map-back, in plain numpy. Mirrors
    merge_equivalents_real::merge() -- 'result = data_group[0]; for i in 1..n:
    result += data_group[i]; result /= n' -- over groups taken in reflection
    order, which is the order a stable sort by packed index leaves them in.
    Do not tidy into data[members].mean(): that is what it must be checked
    against.
    """
    data = np.asarray(data, dtype=np.float64)
    n_segments = int(segment_index.max()) + 1
    means = np.empty(n_segments, dtype=np.float64)
    for g in range(n_segments):
        members = data[segment_index == g]
        acc = members[0]
        for i in range(1, len(members)):
            acc += members[i]
        means[g] = acc / len(members)
    return means[segment_index]


def synthetic_map(n_refl=4000, seed=0, shell_thickness=0.05):
    """A diffuse-like map: strong radial component plus anisotropic structure."""
    rng = np.random.default_rng(seed)
    d_star = rng.uniform(0.02, 0.55, n_refl)
    isotropic = 500.0 * np.exp(-8.0 * d_star)
    anisotropic = rng.normal(0.0, 20.0, n_refl)
    return d_star, isotropic + anisotropic, shell_thickness


def synthetic_partition(n_refl=4000, seed=1, max_multiplicity=16):
    """
    A dense partition with assorted group sizes, as Laue merging produces --
    multiplicity runs to 16 for the 4/mmm Patterson group of P4(1)2(1)2.
    """
    rng = np.random.default_rng(seed)
    multiplicities = []
    while sum(multiplicities) < n_refl:
        multiplicities.append(int(rng.integers(1, max_multiplicity + 1)))
    multiplicities[-1] -= sum(multiplicities) - n_refl
    segment_index = np.repeat(np.arange(len(multiplicities)), multiplicities)
    rng.shuffle(segment_index)  # reflection order is NOT segment order
    # shuffling can empty no segment, but it can reorder the ids; redensify
    _, segment_index = np.unique(segment_index, return_inverse=True)
    return segment_index.ravel().astype(np.int64)


class TestSegmentAverage:
    def test_matches_sequential_merge_exactly(self):
        _, data, _ = synthetic_map()
        segments = synthetic_partition()
        np.testing.assert_array_equal(
            segment_average(data, segments, segment_counts(segments)),
            reference_segment_average(data, segments),
        )

    def test_empty_segment_rejected(self):
        with pytest.raises(ValueError, match="dense"):
            segment_counts(np.array([0, 0, 2, 2]))

class TestAnisoOperator:
    def test_without_segments_matches_subtract_isotropic(self):
        d_star, data, thickness = synthetic_map()
        operator = AnisoOperator(d_star, thickness)
        np.testing.assert_allclose(
            operator(data), subtract_isotropic(data, d_star, thickness),
            rtol=0, atol=1e-9,
        )

    def test_with_segments_matches_the_two_steps_in_sequence(self):
        d_star, data, thickness = synthetic_map()
        segments = synthetic_partition()
        operator = AnisoOperator(d_star, thickness, segment_index=segments)

        expected = reference_segment_average(
            subtract_isotropic(data, d_star, thickness), segments)
        np.testing.assert_allclose(operator(data), expected, rtol=0, atol=1e-9)

    def test_fixed_mask_matches_the_masked_one_shot(self):
        """A fixed valid mask reproduces subtract_isotropic's mask_value."""
        d_star, data, thickness = synthetic_map()
        data = data.copy()
        data[::7] = np.nan
        valid = ~np.isnan(data)

        operator = AnisoOperator(d_star, thickness, valid=valid)
        result = operator(data)

        np.testing.assert_allclose(
            result, subtract_isotropic(data, d_star, thickness),
            rtol=0, atol=1e-9,
        )
        assert np.isnan(result[::7]).all(), "masked points must stay masked"

    def test_too_few_shells_raises(self):
        d_star = np.array([0.10, 0.11, 0.20, 0.21, 0.30, 0.31])
        with pytest.raises(ValueError, match="minimum of 4"):
            AnisoOperator(d_star, 0.1)


class TestReducedForm:
    """reduced=True must be the same operator, stopped one step earlier."""

    def test_gathering_the_reduced_form_gives_the_full_one(self):
        d_star, data, thickness = synthetic_map()
        segments = synthetic_partition()
        full = AnisoOperator(d_star, thickness, segment_index=segments)
        reduced = AnisoOperator(d_star, thickness, segment_index=segments,
                                reduced=True)

        np.testing.assert_allclose(reduced(data)[segments], full(data),
                                   rtol=0, atol=1e-9)

@pytest.mark.skipif(not HAVE_CCTBX, reason="cctbx not available")
class TestParityWithToAniso:
    """
    The comparison that matters: the operator against the live to_aniso(),
    including the cctbx Laue merge it is replacing.
    """

    @staticmethod
    def _fixture():
        xtraj = pytest.importorskip(
            "lunus.command_line.xtraj",
            reason="xtraj needs cctbx, mdtraj, h5py and gemmi")
        from cctbx import crystal, miller
        from cctbx.array_family import flex

        symmetry = crystal.symmetry(
            unit_cell=(34.196, 45.558, 99.044, 90, 90, 90),
            space_group_symbol="P212121")
        miller_set = miller.build_set(crystal_symmetry=symmetry,
                                      anomalous_flag=False, d_min=3.0)
        rng = np.random.default_rng(0)
        d_star = np.sqrt(miller_set.d_star_sq().data().as_numpy_array())
        data = 500.0 * np.exp(-8.0 * d_star) + rng.normal(0.0, 20.0, len(d_star))
        array = miller.array(miller_set=miller_set, data=flex.double(data))
        return xtraj, array, data

    def test_operator_reproduces_to_aniso(self):
        xtraj, array, data = self._fixture()

        expected = xtraj.to_aniso(array, "P212121").data().as_numpy_array()
        operator = xtraj.build_aniso_operator(array, "P212121")
        np.testing.assert_allclose(operator(data), expected, rtol=0, atol=1e-9)

    def test_segments_reproduce_the_merge_grouping_exactly(self):
        """
        The partition, independent of any arithmetic: two reflections share a
        segment exactly when the Laue merge gives them the same value.
        """
        xtraj, array, data = self._fixture()
        from cctbx.array_family import flex

        segments = xtraj.build_aniso_operator(array, "P212121").segment_index
        # A merge of distinct per-group markers: reflections that merge
        # together come back with equal values, and nothing else does.
        marker = array.customized_copy(
            data=flex.double(np.arange(len(data), dtype=np.float64)))
        merged = xtraj.symmetry_average(marker, "P212121").data().as_numpy_array()

        _, from_merge = np.unique(merged, return_inverse=True)
        _, from_segments = np.unique(segments, return_inverse=True)
        np.testing.assert_array_equal(from_merge.ravel(), from_segments.ravel())
