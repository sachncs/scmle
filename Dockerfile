# The scmle image: the released wheel on locked dependencies, run as non-root.
#
# Build it with `python -m scmle.deploy.packaging image --wheel dist/<wheel>`,
# which passes the wheel's name and SHA-256 and the `wheels` build context:
#
#   docker build --build-context wheels=dist \
#     --build-arg WHEEL=scmle-<version>-py3-none-any.whl \
#     --build-arg WHEEL_SHA256=<sha256 of the wheel> -t scmle:<version> .
#
# The base is pinned by the digest of its multi-architecture index (the pin
# in scmle.deploy.base_image.PINNED); the placeholder digest and a bare tag
# are refused. Every dependency and the wheel itself are installed with
# --require-hashes, so a changed file fails the build. This image trains; the
# service folders `scmle package` writes build their own production image.
ARG BASE_IMAGE=python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
RUN case "${BASE_IMAGE}" in \
      *@sha256:0000000000000000000000000000000000000000000000000000000000000000) \
        echo "BASE_IMAGE is the placeholder digest: pin a real one" >&2; exit 1;; \
      *@sha256:*) ;; \
      *) echo "BASE_IMAGE must be pinned by digest" >&2; exit 1;; \
    esac
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HOME=/tmp
ARG WHEEL
ARG WHEEL_SHA256
RUN useradd --system --uid 10001 --user-group --no-create-home \
      --shell /usr/sbin/nologin scmle
COPY deploy/requirements.lock /tmp/install/requirements.lock
COPY --from=wheels ${WHEEL} /tmp/install/
RUN case "${WHEEL}:${WHEEL_SHA256}" in \
      scmle-*.whl:????????????????????????????????????????????????????????????????) ;; \
      *) echo "WHEEL and WHEEL_SHA256 must name the scmle wheel and its hash" >&2; exit 1;; \
    esac \
    && printf '/tmp/install/%s --hash=sha256:%s\n' "${WHEEL}" "${WHEEL_SHA256}" \
      > /tmp/install/wheel.lock \
    && pip install --no-cache-dir --require-hashes --no-deps \
      -r /tmp/install/requirements.lock -r /tmp/install/wheel.lock \
    && pip check \
    && rm -rf /tmp/install
WORKDIR /tmp
USER 10001:10001
CMD ["scmle", "--help"]
