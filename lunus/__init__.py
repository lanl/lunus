from __future__ import division

# The Boost extension is optional AT IMPORT TIME so that the pure-Python
# subpackages -- lunus.md (a self-contained MD engine) and lunus.sf (the
# differentiable structure-factor code, numpy+torch only) -- can be imported
# without a full cctbx module build. Neither uses lunus_ext. Before this,
# `import lunus.md.units` failed on a missing lunus_ext, even though
# __all__ below has always advertised md as the package's public submodule.
#
# Nothing is silently degraded: the extension's names are still bound
# normally when it loads, and when it does not, __getattr__ below turns any
# attempt to reach one into an ImportError naming the actual cause. Code that
# needs Process/LunusDIFFIMAGE therefore still fails loudly, just at first use
# rather than at import of an unrelated subpackage.

_ext_import_error = None

try:
    import boost_adaptbx.boost.python as bp
    ext = bp.import_ext("lunus_ext")
    from lunus_ext import *
except ImportError as e:
    ext = None
    _ext_import_error = e

__all__ = ["md"]

# Top-level names lunus_ext provides. Only these get the explanatory
# ImportError below; everything else raises a plain AttributeError.
#
# That distinction matters more than it looks. __getattr__ fires for EVERY
# missing attribute, including the optional ones that tooling probes for --
# pytest looks up `pytest_plugins` on any module it treats as a package, for
# instance. Raising ImportError for those turns a routine probe into a hard
# collection error, which is exactly what happened before this list existed.
_EXT_NAMES = frozenset(["Process", "LunusDIFFIMAGE", "LunusLAT3D"])


def __getattr__(name):
    """Report the extension's absence at the point of use (PEP 562).

    Only reached for names this module does not define, so it costs nothing
    when lunus_ext loaded successfully.
    """
    if _ext_import_error is not None and (name in _EXT_NAMES
                                          or name.startswith("Lunus")):
        raise ImportError(
            "lunus.%s requires the lunus_ext Boost extension, which is not "
            "importable: %s\n"
            "Build lunus as a cctbx module (place the repo in a cctbx "
            "modules/ directory, then `libtbx.configure lunus; make`). The "
            "pure-Python subpackages lunus.md and lunus.sf do not need it."
            % (name, _ext_import_error)
        )
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
