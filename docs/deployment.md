# Deployment

This document describes how to take the Feedback Intelligence Agent from a local
demo to a deployed service. The manifests under [`deploy/`](../deploy) are
realistic starting points, not a turnkey production platform. Read the
[Scope and honesty](#scope-and-honesty) section before relying on them.

## Health and readiness endpoints

The API exposes two unauthenticated probe endpoints used by every deployment
target below:

| Endpoint | Probe type | Meaning |
|---|---|---|
| `GET /health` | Liveness | The process is up and serving requests. No dependency checks; restart the container if this fails. |
| `GET /ready` | Readiness | The app object (agent, conversation store, job store) was constructed at startup. Reaching the handler proves the app is built and able to serve traffic. |

Both return `200` with a small JSON body (`{"status": "ok"}` and
`{"status": "ready"}`). `/ready` is intentionally honest: this app has no
external dependency to poll in the default local mode, so it does not fake a
database/LLM check. If you later add a hard dependency (a managed vector DB, a
remote LLM), extend `/ready` to verify it.

## Configuration and secrets

All runtime configuration comes from `FEEDBACK_AGENT_*` environment variables
(see [`config.py`](../src/feedback_intelligence_agent/config.py) and the table in
the [README](../README.md#configuration)). The defaults run a fully local,
deterministic pipeline that needs **no API keys**.

Secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are only required when you
switch `FEEDBACK_AGENT_LLM_PROVIDER` away from `local`. They are **never** stored
in the repository:

- Docker Compose prod-like: a local `deploy/.env.prod` file (gitignored).
- ECS Fargate: AWS Secrets Manager, referenced by ARN via `secrets[].valueFrom`.
- Fly.io: `fly secrets set ...`.

## Deployment paths

### 1. Local development

```bash
poetry install
poetry run uvicorn feedback_intelligence_agent.api:create_app --factory --reload
curl http://127.0.0.1:8000/health   # {"status":"ok"}
curl http://127.0.0.1:8000/ready    # {"status":"ready"}
```

### 2. Docker Compose, production-like

[`deploy/docker-compose.prod.yml`](../deploy/docker-compose.prod.yml) runs the
**built image** under gunicorn with uvicorn workers, reads config from an env
file, restarts on failure, and defines a container healthcheck against
`/health`.

```bash
# Build the image (tag it 'latest' to match the compose file).
docker build -t feedback-intelligence-agent:latest .

# Provide configuration via an env file (do NOT commit it).
cp .env.example deploy/.env.prod
# edit deploy/.env.prod as needed

docker compose -f deploy/docker-compose.prod.yml up -d
docker compose -f deploy/docker-compose.prod.yml ps   # shows the healthcheck state
```

Worker count is controlled by `WEB_CONCURRENCY` (default 2). The JSON vector
index is persisted in a named volume so it survives restarts. gunicorn is
installed at container start so the base `Dockerfile` does not need to change.

### 3. AWS ECS Fargate

[`deploy/ecs-fargate-task-definition.json`](../deploy/ecs-fargate-task-definition.json)
is a parameterized task-definition **template**. Replace every placeholder
before registering:

| Placeholder | Example |
|---|---|
| `<AWS_ACCOUNT_ID>` | `123456789012` |
| `<REGION>` | `us-east-1` |
| `<IMAGE_URI>` | `123456789012.dkr.ecr.us-east-1.amazonaws.com/feedback-agent:latest` |
| `<YOUR_FRONTEND_DOMAIN>` | `app.example.com` |

```bash
# Build and push the image to ECR (one-time repo creation omitted).
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
docker build -t <IMAGE_URI> .
docker push <IMAGE_URI>

# Register the task definition (after filling in placeholders).
aws ecs register-task-definition --cli-input-json file://deploy/ecs-fargate-task-definition.json
```

The container `healthCheck` hits `/health`. Secrets are referenced from AWS
Secrets Manager by ARN — create them first; the template only points at them
and contains no secret values. You still need (not included here) an ECS
cluster, a service, an ALB/target group, the two IAM roles, the CloudWatch log
group, and networking.

### 4. Fly.io

[`deploy/fly.toml`](../deploy/fly.toml) builds directly from the repository
Dockerfile and configures both an `/health` (liveness) and `/ready`
(readiness) HTTP check.

```bash
fly launch --no-deploy --copy-config --name <APP_NAME>   # first time only
fly deploy --config deploy/fly.toml --dockerfile Dockerfile

# Optional: switch to a hosted LLM provider with a secret.
fly secrets set ANTHROPIC_API_KEY=sk-ant-... FEEDBACK_AGENT_LLM_PROVIDER=anthropic
```

## Scope and honesty

What these manifests **do** include: a production-like process manager
(gunicorn + uvicorn workers), env-file / secrets-manager based configuration,
restart policies, and liveness/readiness health checks wired to the endpoints
above.

What they **do not** include:

- No Infrastructure-as-Code (no Terraform/CloudFormation/CDK). The ECS template
  is a single task definition; surrounding infrastructure (cluster, service,
  load balancer, IAM roles, networking, log groups) must be created separately.
- No autoscaling, TLS termination config, or CDN.
- No managed datastore by default — the JSON vector store lives on the
  container filesystem / a volume. For multi-instance deployments, switch to
  the Qdrant backend (`FEEDBACK_AGENT_VECTOR_STORE=qdrant`) and point it at a
  shared instance.
- No authentication or rate limiting on the API (see the production
  considerations in [architecture.md](architecture.md)).

The templates require you to fill in account, registry, region, and domain
details. They are meant to demonstrate a credible path to deployment, not to be
applied unmodified to a production account.
