import { useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, LandArea, LandEvidence, ResearchEvent } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import {
  beginInvestigation,
  cancelResearch,
  useInvestigation,
  useInvestigations,
} from "./researchApi";
import { ecologyNames } from "./ecologyRequest";
import { EvidenceView } from "./LandResearch";
import { ResearchArtifactView } from "./ResearchArtifacts";

type Request = components["schemas"]["EcologyRequest"];
interface Operation {
  signature: string;
  id: string;
  key: string;
}
const title = "Ecological context and name review";

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function pendingOperation(key: string): Operation | null {
  try {
    const value: unknown = JSON.parse(read(key) ?? "null");
    if (
      value &&
      typeof value === "object" &&
      "signature" in value &&
      "id" in value &&
      "key" in value &&
      typeof value.signature === "string" &&
      typeof value.id === "string" &&
      typeof value.key === "string"
    )
      return { signature: value.signature, id: value.id, key: value.key };
  } catch {
    /* A damaged browser draft cannot change a server request. */
  }
  return null;
}

export function EcologyResearch({ land, surveyNames }: { land: LandArea; surveyNames: string[] }) {
  const scope = useLandScope(),
    ready = useLandAccessReady(),
    canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const storage = `living-world-land-draft:${encodeURIComponent(scope)}:ecology:${land.id}`;
  const [names, setNames] = useState(() => read(storage + ":names") ?? "");
  const [kingdom, setKingdom] = useState(() => {
    const saved = read(storage + ":kingdom") ?? "";
    return ["Plantae", "Animalia", "Fungi"].includes(saved) ? saved : "";
  });
  const [regions, setRegions] = useState(() => read(storage + ":regions") !== "false"),
    [sites, setSites] = useState(() => read(storage + ":sites") !== "false"),
    [occurrences, setOccurrences] = useState(() => read(storage + ":occurrences") !== "false");
  const [chosen, setChosen] = useState(() => read(storage + ":run"));
  const [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null);
  const [storageError, setStorageError] = useState(false);
  const [evidence, setEvidence] = useState<LandEvidence | null>(null);
  const pending = useRef<Operation | null>(pendingOperation(storage + ":operation"));
  const catalog = useInvestigations(land.id);
  const saved = catalog.data?.filter((item) => item.title === title) ?? [];
  const selected = chosen ?? saved[0]?.id ?? null;
  const detail = useInvestigation(selected, 0, 200);
  // Scope checks also protect against stale browser pointers after switching land/workspace.
  const current = detail.data?.investigation.landId === land.id ? detail.data : undefined;
  const run = current?.runs.find((item) => item.kind === "ecology");
  const running = run?.status === "queued" || run?.status === "running";
  const events = useQuery({
    queryKey: ["land-research", scope, "ecology-events", run?.id],
    enabled: ready && !!run,
    queryFn: async () => {
      const values: ResearchEvent[] = [];
      for (let page = 0; page < 10; page++) {
        const batch = await unwrap(
          api.GET("/api/v1/research/runs/{run_id}/events", {
            params: {
              path: { run_id: run?.id ?? "" },
              query: { after: values.at(-1)?.sequence ?? 0 },
            },
          }),
        );
        values.push(...batch);
        if (batch.length < 200) break;
      }
      return values;
    },
    retry: false,
    refetchInterval: running ? 1500 : false,
  });
  const taxa = ecologyNames(names, kingdom);
  const valid = taxa !== null && (!!taxa?.length || regions || sites || occurrences);
  const store = (key: string, value: string) => {
    try {
      localStorage.setItem(key, value);
    } catch {
      setStorageError(true);
    }
  };
  const updateNames = (text: string) => {
    setNames(text);
    store(storage + ":names", text);
  };
  const choose = (id: string) => {
    setChosen(id);
    setEvidence(null);
    store(storage + ":run", id);
  };
  const start = async () => {
    if (!ready || !canEdit || busy || !valid || !taxa) return;
    setBusy(true);
    setError(null);
    const analysis: Request = {
      dataset: "ecological-context",
      taxa,
      includeEcoregions: regions,
      includeEcologicalSites: sites,
      includeOccurrences: occurrences,
    };
    const signature = JSON.stringify({ revision: land.revision, analysis });
    try {
      let operation = pending.current;
      if (operation?.signature !== signature) {
        const investigation = await beginInvestigation(land, title);
        operation = { signature, id: investigation.id, key: crypto.randomUUID() };
        pending.current = operation;
      }
      store(storage + ":operation", JSON.stringify(operation));
      choose(operation.id);
      await unwrap(
        api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
          params: { path: { investigation_id: operation.id } },
          body: {
            kind: "ecology",
            requestKey: operation.key,
            question: title,
            analysis,
            budget: { maxSteps: 30, maxSeconds: 600, maxWebSearches: 0, maxOutputTokens: 500 },
          },
        }),
      );
      pending.current = null;
      try {
        localStorage.removeItem(storage + ":operation");
      } catch {
        setStorageError(true);
      }
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      await cache.invalidateQueries({ queryKey: ["land-research", scope] });
      setBusy(false);
    }
  };
  const followUp = () => {
    if (!selected) return;
    const context = useLandContext.getState();
    context.selectInvestigation(selected);
    context.setResearchQuestion(
      "Review the saved ecological evidence and any field surveys. What local reference evidence is still needed to define restoration targets?",
    );
    context.setSection("discover");
  };
  return (
    <section className="land-ecology-research" aria-label="Ecological context research">
      <h4>Understand the ecosystem around your land</h4>
      <p>
        Explore regional context, soil-linked reference candidates and biodiversity records. Review
        species names alongside your field observations.
      </p>
      <details className="land-survey-form">
        <summary>Choose sources and review species names</summary>
        <label className="land-check">
          <input
            type="checkbox"
            checked={regions}
            disabled={busy}
            onChange={(e) => {
              setRegions(e.target.checked);
              store(storage + ":regions", String(e.target.checked));
            }}
          />
          EPA regional context · U.S., 2011 edition
        </label>
        <label className="land-check">
          <input
            type="checkbox"
            checked={sites}
            disabled={busy}
            onChange={(e) => {
              setSites(e.target.checked);
              store(storage + ":sites", String(e.target.checked));
            }}
          />
          USDA soil-linked reference candidates · U.S.
        </label>
        <label className="land-check">
          <input
            type="checkbox"
            checked={occurrences}
            disabled={busy}
            onChange={(e) => {
              setOccurrences(e.target.checked);
              store(storage + ":occurrences", String(e.target.checked));
            }}
          />
          GBIF observation sample · global, uneven coverage
        </label>
        <label>
          Scientific names to review (optional)
          <textarea
            aria-label="Scientific names to review"
            rows={4}
            disabled={busy}
            value={names}
            onChange={(e) => updateNames(e.target.value)}
            placeholder={"Quercus alba\nOne scientific name per line"}
          />
        </label>
        <label>
          Kingdom hint
          <select
            aria-label="Kingdom hint"
            value={kingdom}
            disabled={busy}
            onChange={(e) => {
              setKingdom(e.target.value);
              store(storage + ":kingdom", e.target.value);
            }}
          >
            <option value="">Unknown or mixed</option>
            <option value="Plantae">Plants</option>
            <option value="Animalia">Animals</option>
            <option value="Fungi">Fungi</option>
          </select>
        </label>
        {!!surveyNames.length && (
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              updateNames([...new Set(surveyNames.filter((name) => name.trim()))].join("\n"))
            }
          >
            Use names from the displayed survey
          </button>
        )}
        <p className="land-footnote">
          Up to 20 distinct scientific names. Review copied survey labels before looking them up.
          Matching a name does not verify its field identification, native status or suitability for
          restoration.
        </p>
        {!valid && (
          <p role="status">
            Choose at least one source or name. Use at most 20 distinct names, each under 201
            characters.
          </p>
        )}
      </details>
      {error && <p role="alert">{error}</p>}
      {storageError && (
        <p role="status">
          Browser recovery is unavailable. Saved investigations remain in Discover.
        </p>
      )}
      {canEdit && (
        <button
          type="button"
          disabled={!ready || busy || running || !valid}
          onClick={() => void start()}
        >
          {busy ? "Starting ecological research…" : "Research ecological context"}
        </button>
      )}
      <p className="land-footnote">
        Open-data lookups work without an AI connection. Regional references help frame questions;
        site conditions and restoration targets need local evidence.
      </p>
      {!!saved.length && (
        <label>
          Saved ecological research
          <select
            aria-label="Saved ecological research"
            value={selected ?? ""}
            onChange={(e) => choose(e.target.value)}
          >
            {!saved.some((item) => item.id === selected) && (
              <option value={selected ?? ""}>Current investigation</option>
            )}
            {saved.map((item) => (
              <option key={item.id} value={item.id}>
                {new Date(item.createdAt).toLocaleString()} · boundary {item.boundaryRevision}
              </option>
            ))}
          </select>
        </label>
      )}
      {detail.isError && (
        <p role="alert">
          Saved research could not be loaded.{" "}
          <button type="button" onClick={() => void detail.refetch()}>
            Retry ecological results
          </button>
        </p>
      )}
      {run && (
        <div className="land-notice" role="status">
          <strong>Ecological research · {run.status}</strong>
          <p>{run.error}</p>
          {running && canEdit && (
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                setBusy(true);
                void cancelResearch(run.id)
                  .then(() => detail.refetch())
                  .catch((cause: unknown) => setError(describeError(cause)))
                  .finally(() => setBusy(false));
              }}
            >
              Stop ecological research
            </button>
          )}
        </div>
      )}
      {current?.investigation.stale && (
        <p className="land-notice">
          These results use boundary revision {current.investigation.boundaryRevision}. Research
          again to use the current boundary.
        </p>
      )}
      {events.data?.some((event) => event.kind === "source") && (
        <details>
          <summary>Source coverage and lookup activity</summary>
          <ul>
            {events.data
              .filter((event) => event.kind === "source")
              .map((event) => (
                <li key={event.sequence}>
                  <strong>{String(event.payload.status)}</strong> · {String(event.payload.message)}
                </li>
              ))}
          </ul>
        </details>
      )}
      {events.isError && (
        <p role="status">
          Source activity could not be loaded.{" "}
          <button type="button" onClick={() => void events.refetch()}>
            Retry source activity
          </button>
        </p>
      )}
      {current?.findings
        .filter((finding) => finding.runId === run?.id)
        .map((finding) => (
          <article key={finding.id} className="land-finding">
            <h4>{finding.title}</h4>
            <p>{finding.summary}</p>
            <p className="land-footnote">{finding.uncertainty}</p>
            <div className="land-actions">
              {finding.evidenceIds.map((id, index) => {
                const item = current.evidence.find((entry) => entry.id === id);
                return item ? (
                  <button type="button" key={id} onClick={() => setEvidence(item)}>
                    Source {index + 1}
                  </button>
                ) : null;
              })}
            </div>
          </article>
        ))}
      {evidence && <EvidenceView evidence={evidence} onClose={() => setEvidence(null)} />}
      {current?.artifacts
        .filter((artifact) => artifact.runId === run?.id)
        .map((artifact) => (
          <ResearchArtifactView key={artifact.id} artifact={artifact} evidence={current.evidence} />
        ))}
      {run && !running && (
        <button type="button" onClick={followUp}>
          Explore these results with the agent
        </button>
      )}
    </section>
  );
}
