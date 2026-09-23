/**
 * The data console: everything in storage, every run that produced it, and a form that
 * launches more.
 *
 * Three views and one action, per the plan. The action is not a fourth tab — launching a
 * run starts from a capture, so it opens over whichever view you are in and comes back to
 * the Runs table with the new row in it.
 */
import { useEffect, useState } from "react";

import { Boxes, Database, KeyRound, Play, RefreshCw, Rows3 } from "lucide-react";

import { GlassButton, GlassSegmentedControl, GlassTooltipProvider } from "@twin/ui";

import { QueryClient, QueryClientProvider, useQueryClient } from "@tanstack/react-query";

import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

import { CapturesView } from "./CapturesView";
import { NewRun } from "./NewRun";
import { OutputsView } from "./OutputsView";
import { RunsView } from "./RunsView";
import { TokenField } from "./TokenField";

export type AdminView = "captures" | "runs" | "outputs";

const VIEWS: AdminView[] = ["captures", "runs", "outputs"];

function viewFromHash(hash: string): AdminView {
  const name = hash.replace(/^#\/?/, "");
  return VIEWS.find((candidate) => candidate === name) ?? "captures";
}

/**
 * The view lives in the URL fragment so a screen can be linked to.
 *
 * It matters more here than it looks: "the Outputs table on this deployment" is a thing
 * one person sends another, and a console whose state is only in memory cannot be sent.
 */
function useHashView(): [AdminView, (view: AdminView) => void] {
  const [view, setView] = useState<AdminView>(() => viewFromHash(window.location.hash));
  useEffect(() => {
    const onHashChange = () => {
      setView(viewFromHash(window.location.hash));
    };
    window.addEventListener("hashchange", onHashChange);
    return () => {
      window.removeEventListener("hashchange", onHashChange);
    };
  }, []);
  return [
    view,
    (next: AdminView) => {
      window.location.hash = `#/${next}`;
      setView(next);
    },
  ];
}

function Console() {
  const [view, setView] = useHashView();
  const [launching, setLaunching] = useState<string | null>(null);
  const tokenPrompt = useUi((state) => state.writeTokenPrompt);
  const setTokenPrompt = useUi((state) => state.setWriteTokenPrompt);
  const storedToken = useSettings((state) => state.writeToken);
  const [tokenOpen, setTokenOpen] = useState(false);
  const client = useQueryClient();

  // Derived rather than copied into state by an effect. The prompt is raised by the api
  // client's middleware on a 401 and never before it -- a deployment with no
  // API_WRITE_TOKEN leaves writes open, so asking up front would invent a step most runs
  // of this page do not have -- and closing the panel answers the prompt as well as the
  // toggle, which is why both are cleared together below.
  const showToken = tokenOpen || tokenPrompt;
  const closeToken = () => {
    setTokenOpen(false);
    setTokenPrompt(false);
  };

  return (
    <div className="admin">
      <header className="admin__bar">
        <div className="admin__title">
          <Database size={16} aria-hidden="true" />
          <h1>Data console</h1>
        </div>

        <GlassSegmentedControl
          aria-label="Console view"
          value={view}
          onValueChange={setView}
          options={[
            { value: "captures", label: "Captures", icon: <Boxes size={13} aria-hidden="true" /> },
            { value: "runs", label: "Runs", icon: <Rows3 size={13} aria-hidden="true" /> },
            { value: "outputs", label: "Outputs", icon: <Database size={13} aria-hidden="true" /> },
          ]}
        />

        <div className="admin__bar-actions">
          <GlassButton
            size="sm"
            variant="primary"
            leadingIcon={<Play size={13} aria-hidden="true" />}
            onClick={() => {
              setLaunching("");
            }}
            data-testid="new-run"
          >
            New run
          </GlassButton>
          <GlassButton
            size="sm"
            variant="ghost"
            leadingIcon={<RefreshCw size={13} aria-hidden="true" />}
            onClick={() => {
              void client.invalidateQueries({ queryKey: ["admin"] });
            }}
            data-testid="refresh"
          >
            Refresh
          </GlassButton>
          <GlassButton
            size="sm"
            variant={tokenPrompt && !storedToken ? "danger" : "ghost"}
            leadingIcon={<KeyRound size={13} aria-hidden="true" />}
            active={showToken}
            onClick={() => {
              if (showToken) closeToken();
              else setTokenOpen(true);
            }}
            data-testid="token-toggle"
          >
            Write token
          </GlassButton>
          <a className="admin__globe" href="/">
            Open the globe
          </a>
        </div>
      </header>

      {showToken && <TokenField onClose={closeToken} />}

      <main className="admin__body">
        {view === "captures" && (
          <CapturesView
            onLaunch={(captureId) => {
              setLaunching(captureId);
            }}
          />
        )}
        {view === "runs" && <RunsView />}
        {view === "outputs" && <OutputsView />}
      </main>

      {launching !== null && (
        <NewRun
          captureId={launching}
          onClose={() => {
            setLaunching(null);
          }}
          onLaunched={() => {
            setLaunching(null);
            setView("runs");
          }}
        />
      )}
    </div>
  );
}

export function AdminApp() {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider delayDuration={400}>
        <Console />
      </GlassTooltipProvider>
    </QueryClientProvider>
  );
}
