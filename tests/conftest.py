"""Shared pytest fixtures."""

import sys
from types import ModuleType
from unittest.mock import patch

import pytest


def _stub_broken_diffusers_submodules():
    """Pre-load stub modules for diffusers submodules that are missing or broken.

    diffusers 0.36.0 has lazy-import stubs for modules that were never shipped
    (e.g. ``modeling_qwenimage``) or that require optional dependencies not
    installed in this project (e.g. ``sentencepiece`` for Kolors tokenizer).

    ``freezegun`` triggers these lazy imports when it scans every loaded module
    for ``datetime``/``time`` attributes to patch, causing ``RuntimeError`` in
    tests that use ``@freeze_time``.

    By inserting empty stub modules into ``sys.modules`` we prevent the lazy
    import from failing.  This has **no effect** on Scope's runtime — these
    pipelines are never used.
    """
    broken_modules = [
        "diffusers.pipelines.qwenimage.modeling_qwenimage",
        "diffusers.pipelines.pag.pipeline_pag_kolors",
        "diffusers.pipelines.kolors.tokenizer",
        "sentencepiece",
    ]
    for mod_name in broken_modules:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = ModuleType(mod_name)


# Run once at import time so stubs are in place before any test collection
_stub_broken_diffusers_submodules()


@pytest.fixture
def patch_process_functions():
    """Patch functions that spawn processes or terminate pytest."""
    with patch("scope.server.app.os._exit") as mock_exit:
        with patch("scope.server.app.subprocess.Popen") as mock_popen:
            with patch("os.execv") as mock_execv:
                yield {
                    "os_exit": mock_exit,
                    "popen": mock_popen,
                    "execv": mock_execv,
                }
