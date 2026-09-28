"""Inference-time capability modules for the clean-agent experiments.

The modules in this package are deliberately host-owned.  They produce evidence and
memory, but they do not select or execute a robot trajectory.
"""

from .visual_harness import VisualHarness

__all__ = ["VisualHarness"]
