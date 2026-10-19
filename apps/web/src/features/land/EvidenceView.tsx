import { ArrowUpRight, X } from "lucide-react";

import type { LandEvidence } from "@twin/contracts";

import { useLandContext } from "@/state/landContext";

import { LandDocumentViewer } from "./LandDocumentViewer";
import { useLens } from "./surface/lenses";

/** One source behind a finding: what it is, where it came from, its licence and an excerpt. */
export function EvidenceView({
  evidence,
  onClose,
}: {
  evidence: LandEvidence;
  onClose: () => void;
}) {
  return (
    <section className="land-evidence" aria-label="Source evidence">
      <div className="land-place-heading">
        <span className="land-eyebrow">Source evidence</span>
        <button type="button" aria-label="Close source evidence" onClick={onClose}>
          <X size={16} />
        </button>
      </div>
      <h4>{evidence.title}</h4>
      <p>{evidence.relevanceNote}</p>
      <dl>
        <dt>Spatial match</dt>
        <dd>{evidence.spatialRelevance}</dd>
        <dt>Retrieved</dt>
        <dd>{new Date(evidence.retrievedAt).toLocaleDateString()}</dd>
        {evidence.observedAt && (
          <>
            <dt>Observed</dt>
            <dd>{new Date(evidence.observedAt).toLocaleDateString()}</dd>
          </>
        )}
        <dt>License</dt>
        <dd>{evidence.license}</dd>
      </dl>
      <blockquote>{evidence.excerpt}</blockquote>
      <p className="land-footnote">{evidence.attribution}</p>
      {evidence.document && (
        <LandDocumentViewer
          key={`${evidence.document.documentId}:${evidence.document.page}`}
          landId={evidence.document.landId}
          documentId={evidence.document.documentId}
          initialPage={evidence.document.page}
          pinnedHash={evidence.document.sha256}
          pinnedOcrId={evidence.document.ocrId ?? undefined}
        />
      )}
      {evidence.survey && (
        <button
          type="button"
          onClick={() => {
            useLandContext.getState().selectSurvey(evidence.survey?.surveyId ?? null);
            useLens.getState().setLens("ecology");
          }}
        >
          Open cited field survey
        </button>
      )}
      {evidence.inventory && (
        <div className="land-actions">
          <p>
            Saved asset revision {evidence.inventory.revision} · {evidence.inventory.section}
          </p>
          <button
            type="button"
            onClick={() => {
              useLandContext.getState().selectInventory(evidence.inventory?.featureId ?? null);
              useLens.getState().setLens("assets");
            }}
          >
            Open current asset
          </button>
        </div>
      )}
      {evidence.url && (
        <a href={evidence.url} target="_blank" rel="noopener noreferrer">
          Open original source <ArrowUpRight size={14} />
        </a>
      )}
    </section>
  );
}
