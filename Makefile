# Local environment. Targets that need Docker say so; the rest run natively.
PYTHON ?= .venv/bin/python
BASE ?= python:3.12-slim
PLATFORM ?= linux/amd64
WORKDIR ?= runs/local-demo

.PHONY: demo demo-compose image lock lock-check data-check test-local clean-demo

# Train, evaluate, package, serve, predict, monitor and roll back, natively.
demo:
	$(PYTHON) -m scmle.deploy.demo run --workdir $(WORKDIR)

# The same lifecycle in containers (needs Docker Compose v2).
demo-compose: image
	SCMLE_PLATFORM=$(PLATFORM) docker compose up --build --abort-on-container-exit --exit-code-from test-runner --profile test test-runner

# Resolve the base image to a digest, then build the local image.
image:
	$(PYTHON) -m scmle.deploy.base_image resolve --image $(BASE) --output deploy/base-image.lock.json
	docker buildx build --platform $(PLATFORM) \
	  --build-arg BASE_IMAGE="$$($(PYTHON) -m scmle.deploy.base_image argument --lock deploy/base-image.lock.json)" \
	  -f deploy/Dockerfile -t scmle-local:dev --load .

# Export the locked dependencies (with hashes) from uv.lock.
lock:
	scripts/export_locked_requirements.sh

lock-check:
	scripts/export_locked_requirements.sh --check

data-check:
	$(PYTHON) scripts/make_demo_data.py --check
	$(PYTHON) scripts/make_serving_golden.py --check

test-local:
	$(PYTHON) -m pytest -q tests/unit/test_deploy_contract.py tests/unit/test_serving_contract.py tests/unit/test_local_environment.py tests/unit/test_service_hardening.py

clean-demo:
	@echo "delete $(WORKDIR) yourself: it holds the demo's models and registry"
