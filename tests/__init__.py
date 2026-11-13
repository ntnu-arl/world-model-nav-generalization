"""Test package. Isaac Gym must be imported before torch in a process, and test
modules are imported in alphabetical order, so it is imported here first when it
is installed (tests that need it are skipped otherwise)."""

import contextlib

with contextlib.suppress(ImportError):
    import isaacgym  # noqa: F401
