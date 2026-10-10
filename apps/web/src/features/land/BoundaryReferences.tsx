import type { components } from "@twin/contracts";

export function BoundaryReferences({
  sources,
}: {
  sources: components["schemas"]["BoundarySourceReference"][];
}) {
  if (!sources.length) return null;
  return (
    <details className="land-provenance">
      <summary>Drawing guide sources ({sources.length})</summary>
      {sources.map((source, index) => (
        <div className="land-source" key={index}>
          <div>
            <strong>{source.label}</strong>
            {source.recordId && <span>Record {source.recordId}</span>}
            {source.observedAt && (
              <span>Observed {new Date(source.observedAt).toLocaleDateString()}</span>
            )}
            {source.attribution && <small>{source.attribution}</small>}
            {source.url && /^https?:\/\//i.test(source.url) && (
              <a href={source.url} target="_blank" rel="noopener noreferrer">
                Open guide source
              </a>
            )}
          </div>
        </div>
      ))}
    </details>
  );
}
