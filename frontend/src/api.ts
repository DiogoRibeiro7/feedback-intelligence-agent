// Typed client for the Feedback Intelligence Agent FastAPI backend.
// The types mirror the Pydantic schemas in
// src/feedback_intelligence_agent/schemas.py.

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export interface Citation {
  citation_id: number;
  document_id: string;
  chunk_id: string;
  source: string;
  quote: string;
  score: number;
}

export interface GuardrailDecision {
  allowed: boolean;
  reason: string;
  severity: string;
  suggested_response?: string | null;
}

export interface ToolRunRecord {
  tool_name: string;
  status: "ok" | "refused" | "error";
  summary: string;
  output: Record<string, unknown>;
}

export interface AgentAnswer {
  question: string;
  answer: string;
  recommended_actions: string[];
  citations: Citation[];
  route: string;
  confidence: number;
  guardrail?: GuardrailDecision | null;
  tool_run?: ToolRunRecord | null;
  diagnostics: Record<string, unknown>;
}

export interface QueryResponse {
  result: AgentAnswer;
}

export interface QueryRequest {
  question: string;
  top_k?: number;
}

// Payload of the final `metadata` SSE event from POST /query/stream.
export interface StreamMetadata {
  provider: string;
  latency_ms: number;
  route: string;
  confidence: number;
  sources: string[];
  retrieval_scores: number[];
  citations: Citation[];
  recommended_actions: string[];
  guardrail?: GuardrailDecision | null;
}

export class ApiError extends Error {}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      return body.detail;
    }
    return JSON.stringify(body.detail ?? body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

// Non-streaming call: POST /query.
export async function postQuery(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: request.question, top_k: request.top_k ?? 4 }),
  });
  if (!response.ok) {
    throw new ApiError(await readError(response));
  }
  return (await response.json()) as QueryResponse;
}

export interface StreamHandlers {
  onContent: (text: string) => void;
  onMetadata: (metadata: StreamMetadata) => void;
}

// Streaming call: POST /query/stream. Parses the Server-Sent Events stream,
// forwarding each `content` chunk and the single final `metadata` event.
export async function streamQuery(
  request: QueryRequest,
  handlers: StreamHandlers,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: request.question, top_k: request.top_k ?? 4 }),
  });
  if (!response.ok || !response.body) {
    throw new ApiError(await readError(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // SSE events are separated by a blank line. We buffer partial reads and
  // dispatch whole events as they complete.
  const dispatch = (block: string): void => {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) {
        event = line.slice("event:".length).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice("data:".length).trim());
      }
    }
    if (dataLines.length === 0) {
      return;
    }
    const data = JSON.parse(dataLines.join("\n"));
    if (event === "content") {
      handlers.onContent((data as { text: string }).text);
    } else if (event === "metadata") {
      handlers.onMetadata(data as StreamMetadata);
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let separator = buffer.indexOf("\n\n");
    while (separator !== -1) {
      const block = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      if (block.trim()) {
        dispatch(block);
      }
      separator = buffer.indexOf("\n\n");
    }
  }
  if (buffer.trim()) {
    dispatch(buffer);
  }
}
