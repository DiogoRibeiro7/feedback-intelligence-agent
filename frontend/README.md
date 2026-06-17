# Frontend demo

A minimal, professional demo UI for the Feedback Intelligence Agent API, built with
plain **TypeScript + Vite** (no UI framework). It lets you ask a question and
see the grounded answer, the retrieved sources/citations, latency and provider
metadata, and supports an optional **streaming** mode that renders tokens as
they arrive over Server-Sent Events.

## What it shows

- **Question input** and an **answer panel** rendering the agent's response.
- **Retrieved sources panel** listing each citation (`[n] document_id`, source
  channel, retrieval score, and the quoted evidence span).
- **Metadata**: route, confidence, and — in streaming mode — provider and
  `latency_ms`.
- A **streaming toggle**: off calls `POST /query`; on calls `POST /query/stream`
  and appends `content` chunks live, then renders the final `metadata` event.

The typed client in [`src/api.ts`](src/api.ts) mirrors the backend Pydantic
schemas (`QueryResponse`, `AgentAnswer`, `Citation`, `StreamMetadata`).

## Configuration

The backend base URL is configurable via a Vite env var with a sensible default:

| Variable             | Default                 | Description                         |
| -------------------- | ----------------------- | ----------------------------------- |
| `VITE_API_BASE_URL`  | `http://localhost:8000` | Base URL of the FastAPI backend.    |

Copy `.env.example` to `.env` to override it:

```bash
cp .env.example .env
```

## Run locally (dev)

Requires Node.js 20+ and the backend running on `http://localhost:8000`
(`poetry run uvicorn feedback_intelligence_agent.api:create_app --factory --reload`).

```bash
cd frontend
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173). The backend already allows
the Vite dev (`5173`) and preview (`4173`) origins via CORS.

## Build / preview

```bash
npm run build     # type-check (tsc --noEmit) + Vite production build into dist/
npm run preview   # serve the built bundle on http://localhost:4173
```

## Run with Docker Compose

From the repository root, this starts both the backend and the frontend:

```bash
docker compose up --build
```

- Backend API: http://localhost:8000
- Frontend demo: http://localhost:4173

The compose `frontend` service builds [`Dockerfile`](Dockerfile) (a multi-stage
Node build served by `vite preview`) and bakes `VITE_API_BASE_URL=http://localhost:8000`
so the browser calls the backend published on the host.

## Screenshot

No screenshot is committed. To generate one for a portfolio/README:

1. Start the backend and `npm run dev` (or `docker compose up`).
2. Open the UI, run a sample question such as
   _"Why are enterprise customers unhappy with onboarding?"_, optionally toggle
   streaming on.
3. Capture the page (e.g. Windows `Win+Shift+S`, or your browser's
   full-page screenshot) and save it as `docs/frontend.png`.

> _Screenshot placeholder: add `docs/frontend.png` and reference it here._
