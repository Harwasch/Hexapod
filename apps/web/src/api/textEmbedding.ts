import { api, unwrap } from "./client";

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
export const embedQueryText: EncodeText = async (text) => {
  const data = await unwrap(api.GET("/api/v1/text-embeddings", { params: { query: { text } } }));
  return { model: data.model, prompt: data.prompt, vector: Float32Array.from(data.embedding) };
};
