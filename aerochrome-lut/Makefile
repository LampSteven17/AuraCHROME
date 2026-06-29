# Aurachrome — false-color film engine.  Run `make` (or `make help`) for targets.
#
# Typical first run:
#   make setup        # install Poetry (if needed) + Python deps, build the TUI
#   make gpu          # optional: add CUDA acceleration
#   make install      # put the TUI on your PATH (~/.local/bin)
#   make doctor       # verify everything

REPO    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
GO      ?= go
POETRY  ?= poetry
BIN     := aurachrome            # installed command name / launcher script
BINOUT  := tui/aurachrome         # compiled Go binary (gitignored; launcher execs it)
PREFIX  ?= $(HOME)/.local
LDFLAGS := -X main.defaultRepo=$(REPO)

# GPU=auto (default): include the CuPy/CUDA group iff an NVIDIA GPU is detected.
# GPU=1 forces it; GPU=0 skips it. Auto-detect keeps `make deps` from stripping
# the GPU group on a GPU box (Poetry 2.x syncs the env on install).
GPU ?= auto
ifeq ($(GPU),auto)
  GPU_GROUP := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo --with gpu)
else ifeq ($(GPU),1)
  GPU_GROUP := --with gpu
else
  GPU_GROUP :=
endif

.DEFAULT_GOAL := help
.PHONY: help setup bootstrap deps gpu dev all tui run convert install uninstall test doctor clean

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sort \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

setup: bootstrap tui ## One-shot: install Poetry + deps and build the TUI

bootstrap: ## Install Poetry if missing, then core deps
	@command -v $(POETRY) >/dev/null 2>&1 || { \
	  echo "Poetry not found — installing…"; \
	  curl -sSL https://install.python-poetry.org | python3 -; }
	@$(MAKE) deps

deps: ## Install Python deps (auto-adds GPU group when an NVIDIA GPU is present)
	$(POETRY) install $(GPU_GROUP)

gpu: ## Force-install with GPU extras (CuPy + CUDA 12.x libs)
	$(POETRY) install --with gpu

dev: ## Install dev tools (pytest)
	$(POETRY) install --with dev

all: deps tui ## Install deps and build the TUI

tui: ## Build the Go TUI binary (engine path baked in)
	cd $(REPO)/tui && $(GO) build -ldflags "$(LDFLAGS)" -o $(REPO)/$(BINOUT) .
	@echo "built ./$(BINOUT)  (run ./$(BIN) from the repo root)"

run: ## Launch the TUI (the ./aurachrome launcher builds it if needed)
	./$(BIN)

convert: ## Run the engine CLI:  make convert ARGS="-i RAWS -o OUT --gpu"
	$(POETRY) run aurachrome $(ARGS)

install: tui ## Install the TUI into PREFIX/bin (default ~/.local/bin)
	@mkdir -p $(PREFIX)/bin
	install -m 0755 $(REPO)/$(BINOUT) $(PREFIX)/bin/$(BIN)
	@echo "installed $(PREFIX)/bin/$(BIN)  (engine repo baked: $(REPO))"

uninstall: ## Remove the installed TUI binary
	rm -f $(PREFIX)/bin/$(BIN)

test: ## Run Python + Go tests
	$(POETRY) run pytest -q
	cd tui && $(GO) test ./...

doctor: ## Check prerequisites + GPU availability
	@echo "repo:    $(REPO)"
	@printf "python:  "; python3 --version 2>&1 || echo "MISSING"
	@printf "poetry:  "; $(POETRY) --version 2>&1 || echo "MISSING — run 'make bootstrap'"
	@printf "go:      "; $(GO) version 2>&1 || echo "MISSING (needed only for the TUI)"
	@printf "nvidia:  "; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no NVIDIA GPU (CPU mode)"
	@printf "engine:  "; $(POETRY) run aurachrome --help >/dev/null 2>&1 && echo "ok" || echo "not installed — run 'make deps'"
	@printf "backend: "; $(POETRY) run python -c "from aerochrome import backend; print(backend.device_name())" 2>/dev/null || echo "n/a (run 'make deps')"

clean: ## Remove build artifacts
	rm -f $(REPO)/$(BINOUT) $(REPO)/tui/aurachrome-tui $(REPO)/aurachrome-tui
