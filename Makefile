PYTHON ?= .venv/bin/python
PYTEST ?= $(PYTHON) -m pytest

.PHONY: test test-fast test-serial test-smoke

# The config already runs this (-n auto --dist load); the target is here so
# "make test" is the obvious fast path for humans and CI alike.
test:
	$(PYTEST)

# Inner loop: one process, stop at the first failure, show the slowest tests.
test-fast:
	$(PYTEST) -n0 -x -q --durations=10

# Whole suite in one process: readable tracebacks, no worker interleaving.
test-serial:
	$(PYTEST) -n0

# Skip the real-container and end-to-end host/TUI flows.
test-smoke:
	$(PYTEST) -m "not slow"
