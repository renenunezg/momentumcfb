"""Command-line entry point.

After a command returns cleanly the process flushes its streams and exits
without interpreter teardown. On 2026-11-21 a kickoff capture that had already
validated and written its anchors aborted during native-library teardown
("terminate called without an active exception", exit 134), so the workflow
skipped publishing anchors and closing lines for 16 games. Nothing in this
package registers exit handlers, so skipping teardown loses nothing. Failures
keep the normal exit path and exit code.
"""

import logging
import os
import sys

from backend.cli import main

main()
logging.shutdown()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
