"""The auto-tuning rules in tune.py.

These pin the DECISIONS, not the constants: a rule that silently stops firing
is the failure mode that matters, since its whole purpose is to spare the user
an experiment they will not know to repeat.
"""

import pytest

from lunus.sf.tune import (
    DeviceInfo,
    MAX_PAIRS_CPU,
    describe_device,
    estimated_peak_bytes,
    memory_warning,
    recommended_max_pairs,
)

CPU = DeviceInfo(kind="cpu")
MPS = DeviceInfo(kind="mps")
# An 80 GB card with a 50 MB L2, most of it free.
CUDA = DeviceInfo(kind="cuda", l2_bytes=50 << 20, free_bytes=70 * 10**9,
                  total_bytes=80 * 10**9, name="Test GPU")


class TestMaxPairs:
    def test_cpu_keeps_the_measured_default(self):
        pairs, why = recommended_max_pairs(CPU)
        assert pairs == MAX_PAIRS_CPU
        assert "measured" in why

    def test_cuda_sizes_to_l2(self):
        pairs, why = recommended_max_pairs(CUDA)
        assert pairs == (50 << 20) // 4
        assert "L2" in why
        # The point of the rule: a GPU gets a bigger budget than the CPU one.
        assert pairs > MAX_PAIRS_CPU

    def test_unknown_l2_falls_back_rather_than_guessing(self):
        info = DeviceInfo(kind="cuda", l2_bytes=None, free_bytes=70 * 10**9)
        pairs, why = recommended_max_pairs(info)
        assert pairs == MAX_PAIRS_CPU
        assert "did not report" in why

    def test_small_free_memory_caps_the_budget(self):
        info = DeviceInfo(kind="cuda", l2_bytes=96 << 20, free_bytes=200 << 20)
        pairs, why = recommended_max_pairs(info)
        assert pairs < (96 << 20) // 4
        assert "capped" in why or "floored" in why

    def test_never_below_the_measured_default(self):
        info = DeviceInfo(kind="cuda", l2_bytes=1 << 20, free_bytes=1 << 20)
        pairs, _ = recommended_max_pairs(info)
        assert pairs == MAX_PAIRS_CPU

    def test_cap_cannot_raise_the_budget(self):
        """Plentiful memory must not push the budget past the cache rule."""
        lean = DeviceInfo(kind="cuda", l2_bytes=8 << 20, free_bytes=10**12)
        pairs, _ = recommended_max_pairs(lean)
        assert pairs == max((8 << 20) // 4, MAX_PAIRS_CPU)


class TestMemory:
    def test_peak_grows_as_dmin_cubed(self):
        """Halving the voxel size is 8x the grid, which is the thing that
        makes a finer run fail where a coarser one fits.

        Atoms and pairs are zeroed so this isolates the grid terms; the ratio
        lands just under 8 because the rfft's last axis is nw//2 + 1, whose
        +1 is proportionally larger on the coarser grid.
        """
        coarse = estimated_peak_bytes(0, (100, 100, 100), 0)
        fine = estimated_peak_bytes(0, (200, 200, 200), 0)
        assert fine / coarse == pytest.approx(8.0, rel=0.01)
        assert fine / coarse < 8.0

    def test_pair_buffers_and_atoms_are_counted(self):
        """The grid is not the only term: a large pair budget shows up too."""
        bare = estimated_peak_bytes(0, (64, 64, 64), 0)
        with_pairs = estimated_peak_bytes(0, (64, 64, 64), MAX_PAIRS_CPU)
        assert with_pairs > bare
        assert estimated_peak_bytes(10_000, (64, 64, 64), 0) > bare

    def test_quiet_when_it_fits(self):
        assert memory_warning(10_000, (100, 100, 100), MAX_PAIRS_CPU, CUDA) is None

    def test_warns_when_close(self):
        tight = DeviceInfo(kind="cuda", l2_bytes=50 << 20, free_bytes=1 << 30,
                           name="Small GPU")
        msg = memory_warning(100_000, (512, 512, 512), MAX_PAIRS_CPU, tight)
        assert msg is not None and "GB free" in msg

    def test_silent_without_a_free_memory_reading(self):
        assert memory_warning(10_000, (512, 512, 512), MAX_PAIRS_CPU, CPU) is None


class TestDescribeDevice:
    @pytest.mark.parametrize("dev,kind", [
        ("cpu", "cpu"), ("mps", "mps"), ("cuda", "cuda"), ("cuda:1", "cuda"),
    ])
    def test_kind_parsing(self, dev, kind):
        assert describe_device(dev).kind == kind

    def test_non_cuda_needs_no_torch(self):
        """Must not import torch to describe a CPU run."""
        info = describe_device("cpu", torch_module=object())
        assert info == DeviceInfo(kind="cpu")

    def test_cuda_failure_degrades_to_kind_only(self):
        class Broken:
            class cuda:
                @staticmethod
                def get_device_properties(_):
                    raise RuntimeError("no driver")
        info = describe_device("cuda", torch_module=Broken())
        assert info.kind == "cuda" and info.l2_bytes is None
        # and the rules must still return something usable
        assert recommended_max_pairs(info)[0] == MAX_PAIRS_CPU
