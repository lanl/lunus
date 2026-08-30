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
                 "mean_and_diffuse", "splat_density", "orth_matrix",
                 "it92_coefficients", "symmetrize_sum"]:
        assert hasattr(sf, name), f"lunus.sf does not expose {name}"


def test_sf_unknown_attribute_raises_attribute_error():
    import lunus.sf as sf

    with pytest.raises(AttributeError):
        sf.definitely_not_a_real_name


def test_bare_sf_import_does_not_pull_in_torch():
    """The lazy export table exists so that `import lunus.sf` stays cheap for
    callers who only wanted lunus.md. Run in a subprocess because torch is
    almost certainly already imported in this one.
    """
    import os
    import subprocess
    import sys

    # The child gets a fresh interpreter, so hand it the directory `lunus` was
    # actually imported from -- which may be a clone on sys.path rather than an
    # installed package.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(lunus.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    code = "import sys; import lunus.sf; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", (
        "importing lunus.sf pulled in torch; the export table should be lazy"
    )
