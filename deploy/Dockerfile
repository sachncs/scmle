# The local environment's image: scmle from source on locked dependencies.
# Build with `make image`, which resolves BASE_IMAGE to a digest first. The
# default below is a placeholder on purpose: the build refuses to run on it.
ARG BASE_IMAGE=python:3.12-slim@sha256:0000000000000000000000000000000000000000000000000000000000000000
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
RUN case "${BASE_IMAGE}" in \
      *@sha256:0000000000000000000000000000000000000000000000000000000000000000) \
        echo "BASE_IMAGE is the placeholder digest: build with 'make image'" >&2; exit 1;; \
      *@sha256:*) ;; \
      *) echo "BASE_IMAGE must be pinned by digest" >&2; exit 1;; \
    esac
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app
RUN useradd --system --uid 10001 --user-group --no-create-home \
      --shell /usr/sbin/nologin scmle
WORKDIR /app
COPY deploy/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --no-deps \
      -r /tmp/requirements.lock && pip check && rm /tmp/requirements.lock
COPY scmle ./scmle
COPY scripts ./scripts
COPY deploy/local ./deploy/local
USER 10001:10001
CMD ["python", "-m", "scmle.deploy.demo", "--help"]
