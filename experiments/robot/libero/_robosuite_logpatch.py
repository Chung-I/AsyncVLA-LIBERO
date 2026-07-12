"""Redirect robosuite's hardcoded ``/tmp/robosuite.log`` file logger to a per-user path.

robosuite's ``utils/log_utils.py`` opens ``logging.FileHandler("/tmp/robosuite.log")`` at
import time (module level). On a shared cluster, ``/tmp/robosuite.log`` is frequently owned
by another user, so ``import robosuite`` (transitively, ``from libero.libero import ...``)
raises ``PermissionError: [Errno 13] Permission denied: '/tmp/robosuite.log'`` and the whole
process dies before a single episode runs.

Importing THIS module before any robosuite/LIBERO import globally swaps ``logging.FileHandler``
for a subclass that rewrites just that one hardcoded path to ``~/.robosuite.log`` (always
writable by the current user); every other ``FileHandler`` call is untouched. Keeping it as a
``FileHandler`` subclass preserves ``isinstance`` checks elsewhere.
"""

import logging
import os

_PER_USER_ROBOSUITE_LOG = os.path.join(os.path.expanduser("~"), ".robosuite.log")


class _RedirectingFileHandler(logging.FileHandler):
    def __init__(self, filename, *args, **kwargs):
        if str(filename) == "/tmp/robosuite.log":
            filename = _PER_USER_ROBOSUITE_LOG
        super().__init__(filename, *args, **kwargs)


# Idempotent: only wrap once even if imported from multiple entrypoints.
if getattr(logging.FileHandler, "_robosuite_logpatch_applied", False) is False:
    _RedirectingFileHandler._robosuite_logpatch_applied = True
    logging.FileHandler = _RedirectingFileHandler
