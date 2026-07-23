# iMem — one-command dev orchestration.
#
# The stack spans two runtimes: Qdrant + the frontend run in Docker (compose),
# while the backend API runs on the host — it needs the host GPU/MPS, read
# access to arbitrary indexed image paths, and (on macOS) TCC folder
# permissions, so it can't be containerized. These targets drive both worlds
# from one place.
#
# Override variables on the command line, e.g.:
#   make install PYTHON=/opt/homebrew/anaconda3/envs/imem/bin/python
#   make serve   IMEM_BASE_DIR=/some/other/root

# Python used for install/tests. Defaults to `python` on PATH — so with the
# imem conda env activated, this already resolves to the env's interpreter.
PYTHON ?= python

# Base dir the API resolves *relative* (legacy) image paths against. Defaults to
# your home dir, where older collections were indexed from.
IMEM_BASE_DIR ?= $(HOME)

.DEFAULT_GOAL := help
.PHONY: help install build up serve dev down test

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

install: ## Install the imem package (backend + tests) into the active env
	$(PYTHON) -m pip install -e ".[api,dev]"

build: ## Build the frontend Docker image
	docker compose build frontend

up: ## Start the Qdrant + frontend containers (builds the frontend if needed)
	docker compose up -d

serve: ## Run the backend API on the host (foreground)
	IMEM_BASE_DIR="$(IMEM_BASE_DIR)" imem serve

dev: up install serve ## Bring the whole stack up, install, then run the backend

down: ## Stop and remove the containers
	docker compose down

test: ## Run the test suite
	$(PYTHON) -m pytest
