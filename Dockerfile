# Single-container image for ECS Fargate (or any Docker host). Serves the web app
# and calls Claude. Secrets (ANTHROPIC_API_KEY, RRN_ORIGIN_SECRET) are injected at
# runtime, never baked in. Reports and uploads go to S3 when RRN_S3_BUCKET is set.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so code changes don't rebuild this layer.
COPY pyproject.toml README.md ./
RUN mkdir -p src/researchreadernerd && touch src/researchreadernerd/__init__.py \
    && pip install ".[aws]" && pip uninstall -y researchreadernerd

COPY src ./src
RUN pip install --no-deps .

# Deployment identity, reported by /health so a deploy can be verified:
#   docker build --build-arg BUILD_SHA=$(git rev-parse --short HEAD) \
#                --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) .
ARG BUILD_SHA=""
ARG BUILD_DATE=""
ENV BUILD_SHA=$BUILD_SHA \
    BUILD_DATE=$BUILD_DATE

# Run as a non-root user.
RUN useradd --create-home app
USER app

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "rrn-web --host 0.0.0.0 --port ${PORT:-8080}"]
