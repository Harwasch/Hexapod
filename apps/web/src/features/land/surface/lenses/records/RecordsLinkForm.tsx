import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { components } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandScope } from "@/state/landIdentity";

import { useLandDocuments } from "./recordsApi";
import { RELATIONS } from "./recordsModel";

type Link = components["schemas"]["DocumentLinkCreate"];

/** Records a connection from a page of one document to a page of another. */
export function RecordsLinkForm({
  landId,
  documentId,
  page,
  onDone,
}: {
  landId: string;
  documentId: string;
  page: number;
  onDone: () => void;
}) {
  const scope = useLandScope();
  const cache = useQueryClient();
  const documents = useLandDocuments(landId, 0);
  const [link, setLink] = useState<Link>(() => ({
    requestKey: crypto.randomUUID(),
    fromDocumentId: documentId,
    fromPage: page,
    toDocumentId: "",
    toPage: 1,
    relation: "related",
    basis: "",
  }));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const save = async () => {
    const request = new AbortController();
    controller.current?.abort();
    controller.current = request;
    setBusy(true);
    setError(null);
    try {
      await unwrap(
        api.POST("/api/v1/land/{land_id}/documents/links", {
          params: { path: { land_id: landId } },
          body: link,
          signal: request.signal,
        }),
      );
      await cache.invalidateQueries({ queryKey: ["land-document-links", scope, landId] });
      if (!request.signal.aborted) onDone();
    } catch (cause) {
      if (!request.signal.aborted)
        setError(`The connection could not be saved. ${describeError(cause)}`);
    } finally {
      if (!request.signal.aborted) setBusy(false);
    }
  };
  const others = (documents.data ?? []).filter(
    (document) => document.pageCount > 0 && document.id !== documentId,
  );
  return (
    <form
      className="records-form"
      aria-label="Relate this record to another"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <fieldset disabled={busy} className="records-form__fields">
        <legend className="records-subhead">Relate this record to another</legend>
        <div className="records-grid">
          <label className="land-field">
            Source page
            <input
              className="glass-input"
              required
              type="number"
              min={1}
              max={500}
              value={link.fromPage}
              onChange={(event) => setLink({ ...link, fromPage: Number(event.target.value) })}
            />
          </label>
          <label className="land-field">
            Relationship
            <select
              className="glass-input"
              value={link.relation}
              onChange={(event) =>
                setLink({ ...link, relation: event.target.value as Link["relation"] })
              }
            >
              {RELATIONS.map((relation) => (
                <option key={relation.value} value={relation.value}>
                  {relation.label}
                </option>
              ))}
            </select>
          </label>
          <label className="land-field">
            Related record
            <select
              className="glass-input"
              required
              value={link.toDocumentId}
              onChange={(event) => setLink({ ...link, toDocumentId: event.target.value })}
            >
              <option value="">Choose a record</option>
              {others.map((document) => (
                <option key={document.id} value={document.id}>
                  {document.title}
                </option>
              ))}
            </select>
          </label>
          <label className="land-field">
            Related page
            <input
              className="glass-input"
              required
              type="number"
              min={1}
              max={500}
              value={link.toPage}
              onChange={(event) => setLink({ ...link, toPage: Number(event.target.value) })}
            />
          </label>
        </div>
        <label className="land-field">
          What shows they are connected?
          <textarea
            className="glass-input"
            required
            maxLength={3000}
            value={link.basis}
            onChange={(event) => setLink({ ...link, basis: event.target.value })}
          />
        </label>
        <label className="land-field">
          Date it took effect, if known
          <input
            className="glass-input"
            type="date"
            value={link.effectiveDate ?? ""}
            onChange={(event) => setLink({ ...link, effectiveDate: event.target.value || null })}
          />
        </label>
        <p className="land-muted">A connection is a lead to check, not proof of legal effect.</p>
        {error && (
          <p className="land-alert" role="alert">
            {error}
          </p>
        )}
        <div className="land-row">
          <GlassButton type="submit" size="sm" variant="primary" loading={busy}>
            Save relationship
          </GlassButton>
          <GlassButton size="sm" variant="ghost" onClick={onDone}>
            Cancel
          </GlassButton>
        </div>
      </fieldset>
    </form>
  );
}
