import { api, ApiError, unwrap } from "./client";

/** A search query embedded by the API with the model scan objects were described with. */
export interface QueryEmbedding {
  /** As `instances.json`'s `embedding.model` names it; only that scan's rows are comparable. */
  model: string;
  /** What was embedded: the query inside the tags' template ("a photo of a spool."). */
  prompt: string;
  vector: Float32Array;
}

export type EncodeText = (text: string) => Promise<QueryEmbedding>;

/**
 * `GET /api/v1/text-embeddings` (apps/api/app/services/text_encoder.py): SigLIP 2's text
 * tower on the API's CPU. The first call on a machine loads the model (~1.3 s), later ones
 * take ~140 ms; a deployment without the encoder answers 503, and search stays on tags.
 */
/**
 * What the API embeds of a query: lower-cased, a leading "a photo of" and article dropped
 * (`query_label` in apps/api/app/services/text_encoder.py, which this mirrors). Empty when
 * nothing is left, which the API would refuse.
 */
export function queryLabel(text: string): string {
  const label = text
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .join(" ")
    .replace(/[ .]+$/, "");
  return label.replace(/^(?:a photo of )?(?:(?:a|an|the)(?:\s+|$))?/, "").trim();
}

/** The API's longest query (`MAX_QUERY_CHARS`); a longer one is refused with a 422. */
export const MAX_QUERY_CHARS = 200;

/** Whether the API would embed `text` rather than refuse it (422). */
export function embeddable(text: string): boolean {
  return text.trim().length > 0 && text.length <= MAX_QUERY_CHARS && queryLabel(text) !== "";
}

/**
 * Whether a failed request means the encoder is down (back off) rather than that this query
 * was refused: a 5xx (503 is "no encoder here") or no answer at all (offline, a network
 * error). A 4xx is about the query; the next one may well be fine.
 */
export function encoderDown(error: unknown): boolean {
  return !(error instanceof ApiError) || error.status >= 500;
}

export const embedQueryText: EncodeText = async (text) => {
  const data = await unwrap(api.GET("/api/v1/text-embeddings", { params: { query: { text } } }));
  return { model: data.model, prompt: data.prompt, vector: Float32Array.from(data.embedding) };
};
