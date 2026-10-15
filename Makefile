help:  ## Show the targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

test: ## Run the tests (fast ones only)
	pytest -m "not slow"

test-all: ## Run every test, including the end-to-end workflow
	pytest

format: ## Lint and format (pre-commit hooks)
	pre-commit run -a

clean: ## Remove caches and build artefacts
	rm -rf dist build .pytest_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
