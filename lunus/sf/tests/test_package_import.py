"""
Checks that importing lunus and lunus.sf behaves, with and without the
lunus_ext Boost extension present.

The subtle requirement is the one in test_unknown_attribute_raises_attribute_error:
lunus/__init__.py defines a module __getattr__ so that `lunus.Process` explains
itself when the extension is missing, and __getattr__ fires for EVERY missing
attribute -- including the optional ones tooling probes for. pytest looks up
`pytest_plugins` on any module it treats as a package; an earlier version
raised ImportError for that, which turned a routine probe into a collection
error for the whole suite. Probes must get AttributeError.
"""

import importlib
import sys

import pytest

import lunus


HAVE_EXT = lunus.ext is not None


def test_lunus_imports_without_the_extension():
    """Importing the package must never depend on lunus_ext."""
    importlib.reload(lunus)
    assert lunus is not None


def test_pure_python_subpackages_are_importable():
    """lunus.md and lunus.sf use nothing from the extension, so they must
    import whether or not it was built."""
    importlib.import_module("lunus.md.units")
    importlib.import_module("lunus.sf")


@pytest.mark.parametrize(
    "name",
    ["pytest_plugins", "__bases__", "__test__", "__wrapped__", "_pytest",
     "__all_subclasses__", "nonexistent_thing"],
)
def test_unknown_attribute_raises_attribute_error(name):
    """Probes for optional attributes must raise AttributeError.

    ImportError here breaks tools that legitimately ask whether a module
    defines something optional.
    """
    with pytest.raises(AttributeError):
        getattr(lunus, name)


@pytest.mark.skipif(HAVE_EXT, reason="lunus_ext is built; nothing to explain")
@pytest.mark.parametrize("name", ["Process", "LunusDIFFIMAGE", "LunusLAT3D"])
def test_extension_names_explain_themselves(name):
    """Without the extension, its own names raise ImportError naming the
    cause and the fix -- not a bare AttributeError."""
    with pytest.raises(ImportError, match="requires the lunus_ext"):
        getattr(lunus, name)


@pytest.mark.skipif(not HAVE_EXT, reason="lunus_ext is not built here")
@pytest.mark.parametrize("name", ["Process", "LunusDIFFIMAGE"])
def test_extension_names_resolve_when_built(name):
    assert getattr(lunus, name) is not None


def test_sf_public_api_is_importable():
    """The names README advertises must actually resolve from the package
    root, not just from their submodules."""
    import lunus.sf as sf

    for name in ["structure_factors_batch", "structure_factors_one_config",
                 "mean_and_diffuse", "compute_density", "orth_matrix",
                 "it92_coefficients", "symmetrize_sum"]:
        assert hasattr(sf, name), f"lunus.sf does not expose {name}"


def test_deprecated_splat_density_alias_warns():
    """The old name still resolves from the package root and says it is old."""
    import warnings

    pytest.importorskip("torch")
    import lunus.sf as sf

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError):        # no arguments: warns, then forwards
            sf.splat_density()
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_sf_unknown_attribute_raises_attribute_error():
    import lunus.sf as sf

    with pytest.raises(AttributeError):
        sf.definitely_not_a_real_name


def run_in_fresh_interpreter(code):
    """Run `code` in a child interpreter that imports the same `lunus` this
    process did -- which may be a clone on sys.path rather than an installed
    package -- and return the CompletedProcess after checking it succeeded."""
    import os
    import subprocess

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(lunus.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out


def test_bare_sf_import_does_not_pull_in_torch():
    """The lazy export table exists so that `import lunus.sf` stays cheap for
    callers who only wanted lunus.md. Run in a subprocess because torch is
    almost certainly already imported in this one.
    """
    out = run_in_fresh_interpreter(
        "import sys; import lunus.sf; print('torch' in sys.modules)")
    assert out.stdout.strip() == "False", (
        "importing lunus.sf pulled in torch; the export table should be lazy"
    )


# Stands in for a boost_adaptbx whose import_ext fails the way the real one
# does: RTLD_GLOBAL set for the load, then an ImportError with the flags left
# set. Lets the dlopen test below exercise the leak where cctbx is absent, in
# which case the real `import boost_adaptbx` fails before touching the flags
# and the test would pass whether or not lunus restores them.
LEAKY_BOOST_ADAPTBX = """
import os, sys, types
def import_ext(name, optional=False):
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
    raise ImportError("stub: %s not built" % name)
for modname in ("boost_adaptbx", "boost_adaptbx.boost", "boost_adaptbx.boost.python"):
    sys.modules[modname] = types.ModuleType(modname)
sys.modules["boost_adaptbx.boost.python"].import_ext = import_ext
"""


@pytest.mark.skipif(not hasattr(sys, "getdlopenflags"),
                    reason="no dlopen flags on this platform")
@pytest.mark.parametrize("boost", ["installed", "leaky_stub"])
def test_import_leaves_dlopen_flags_unchanged(boost):
    """`import lunus` must not leave RTLD_GLOBAL set. boost_adaptbx's
    import_ext sets it to load an extension and restores the old flags only
    when the import succeeds, so without lunus_ext every import of lunus used
    to leave the process loading all later shared libraries globally -- which
    makes every torch.compile CPU kernel after the first call the first one's
    `kernel`, whatever it was compiled for (silently wrong results; wrong
    densities, then a segfault, in test_torch_pipeline.py). Run in a
    subprocess: the flags are process state, and this one has imported lunus.
    "installed" uses whatever boost_adaptbx (if any) this environment has;
    "leaky_stub" replaces it with one that always leaks, so the regression is
    caught without cctbx too.
    """
    code = ("import sys; before = sys.getdlopenflags(); import lunus; "
            "print(before, sys.getdlopenflags(), lunus.ext is not None)")
    if boost == "leaky_stub":
        code = LEAKY_BOOST_ADAPTBX + code
    out = run_in_fresh_interpreter(code)
    # Last line only: anything the import itself printed comes before it.
    before, after, have_ext = out.stdout.splitlines()[-1].split()
    assert after == before, (
        "import lunus changed the dlopen flags from %#x to %#x (lunus_ext built: %s); "
        "RTLD_GLOBAL left set breaks torch.compile" % (int(before), int(after), have_ext))
