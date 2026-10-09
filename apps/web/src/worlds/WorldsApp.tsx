import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import {
  ArrowRight,
  AudioLines,
  Box,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Copy,
  Download,
  FlaskConical,
  FolderOpen,
  Gamepad2,
  Globe2,
  Home,
  ImagePlus,
  Layers3,
  Loader2,
  Monitor,
  Play,
  Plus,
  Radio,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  UsersRound,
  Video,
  WandSparkles,
  X,
} from "lucide-react";
import { SceneArt } from "./SceneArt";
import { assessCompute, useComputeReadiness } from "./core/readiness";
import { WorkerManager } from "./WorkerManager";
import { ComputeInventory } from "./ComputeInventory";
import { BillingPanel } from "./BillingPanel";
import { MediaConditioning } from "./MediaConditioning";
import { ReadinessDetails, ReadinessRefresh } from "./ComputeReadiness";
import Player from "./Player";
import ExperimentLab from "./experiments/ExperimentLab";
import { CharacterStudio, ReplayHighlights, CustomizationPanel } from "./intelligence";
import { ReconstructionPanel } from "./reconstruction/ReconstructionPanel";
import {
  MODELS,
  PROVIDERS,
  getModel,
  worldStore,
  getSettings,
  saveSettings,
  getApiToken,
  setApiToken,
  newId,
  type AppSettings,
  type Character,
  type MediaAsset,
  type ProviderId,
  type Replay,
  type WorldProject,
  type WorldScene,
  type Reconstruction,
  type BenchmarkResult,
  createWorldApi,
  validateApiBaseUrl,
} from "./core";
import "./worlds.css";

type Area = "home" | "create" | "worlds" | "characters" | "replays" | "3d" | "lab" | "settings";
interface Template {
  id: string;
  name: string;
  eyebrow: string;
  description: string;
  prompt: string;
  kind: string;
  tags: string[];
}
const TEMPLATES: [Template, ...Template[]] = [
  {
    id: "redwood",
    name: "Redwood Signal",
    eyebrow: "GET LOST. FIND SOMETHING.",
    description: "Somewhere beyond the trees, a signal is calling.",
    prompt:
      "First-person exploration of a vast old-growth redwood forest at dusk. Mist drifts between enormous trunks, ferns cover the forest floor, and a distant abandoned radio transmitter glows through the canopy. Follow a winding navigable trail toward the signal. Cinematic natural light, realistic scale, subtle wind.",
    kind: "redwood",
    tags: ["Exploration", "Atmospheric"],
  },
  {
    id: "road",
    name: "Infinite Road",
    eyebrow: "THE JOURNEY IS THE WORLD.",
    description: "One road. A thousand places it could take you.",
    prompt:
      "First-person driving along an endless two-lane mountain road at sunset. Layered alpine peaks, sweeping turns and warm light on the asphalt. The landscape unfolds naturally ahead. Cinematic realistic scale and coherent forward movement.",
    kind: "road",
    tags: ["Driving", "Open-ended"],
  },
  {
    id: "alien",
    name: "Alien Expedition",
    eyebrow: "FAR FROM ANYTHING FAMILIAR.",
    description: "Leave the familiar behind. Follow your curiosity.",
    prompt:
      "First-person expedition across an unfamiliar alien planet, towering mineral formations, luminous native plants, a ringed planet above the horizon. A traversable trail winds between strange landmarks. Muted cinematic colors, atmospheric depth and realistic scale.",
    kind: "alien",
    tags: ["Sci-fi", "Discovery"],
  },
  {
    id: "dungeon",
    name: "Dungeon Run",
    eyebrow: "THERE IS ALWAYS ANOTHER DOOR.",
    description: "Ancient halls. Unwritten legends. Your next move.",
    prompt:
      "First-person exploration of an ancient stone dungeon, torchlit arches, worn stairs, mysterious doors and a faint golden light at the end of the corridor. A cinematic fantasy adventure with navigable corridors.",
    kind: "dungeon",
    tags: ["Fantasy", "Adventure"],
  },
  {
    id: "horror",
    name: "Horror House",
    eyebrow: "YOU HEARD THAT TOO.",
    description: "A light in the window. A house that remembers.",
    prompt:
      "First-person exploration of a deserted mansion at the edge of a fog-covered forest at night. A flickering window draws you toward the entrance. Slow atmospheric suspense, weathered timber and moonlit overgrown paths.",
    kind: "horror",
    tags: ["Horror", "Atmospheric"],
  },
  {
    id: "dream",
    name: "Dream World",
    eyebrow: "REALITY IS A SUGGESTION.",
    description: "An impossible place, waiting for a new idea.",
    prompt:
      "First-person exploration of a surreal dreamscape. Floating stone islands hover above a still ocean, a luminous doorway stands under an enormous pale moon, soft lavender clouds drift slowly through the scene. Ethereal cinematic lighting.",
    kind: "dream",
    tags: ["Surreal", "Creative"],
  },
  {
    id: "photo",
    name: "Photo Portal",
    eyebrow: "STEP INTO A MOMENT.",
    description: "Start with an image. Discover what lies beyond it.",
    prompt:
      "Continue the environment in the reference image as an immersive first-person scene. Preserve its visual style, lighting and spatial layout while exploring gradually beyond the original view.",
    kind: "dream",
    tags: ["Image input", "Conditioning"],
  },
  {
    id: "video",
    name: "Video Continuation",
    eyebrow: "AFTER THE LAST FRAME.",
    description: "The end of a clip can be the beginning of a world.",
    prompt:
      "Continue the uploaded video beyond its final frame, maintaining the camera trajectory, environment and temporal continuity.",
    kind: "road",
    tags: ["Video input", "Experimental"],
  },
];
const NAV: { id: Area; label: string; icon: typeof Home }[] = [
  { id: "home", label: "Discover", icon: Home },
  { id: "worlds", label: "My worlds", icon: FolderOpen },
  { id: "characters", label: "Characters", icon: UsersRound },
  { id: "replays", label: "Replays", icon: Video },
  { id: "3d", label: "3D worlds", icon: Box },
];
function relativeTime(time: number) {
  const d = Math.max(0, Math.floor((Date.now() - time) / 86400000));
  return d === 0 ? "Today" : d === 1 ? "Yesterday" : `${d} days ago`;
}
function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function AssetImage({ id, alt, className = "" }: { id?: string; alt: string; className?: string }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    let active = true;
    let objectUrl = "";
    if (id)
      void worldStore.getBlob(id).then((blob) => {
        if (blob && active) {
          objectUrl = URL.createObjectURL(blob);
          setUrl(objectUrl);
        }
      });
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [id]);
  return url ? <img src={url} alt={alt} className={className} /> : null;
}
function EmptyState({
  icon: Icon,
  title,
  children,
  action,
}: {
  icon: typeof Home;
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="w-empty">
      <div className="w-empty-icon">
        <Icon size={30} />
      </div>
      <h2>{title}</h2>
      <p>{children}</p>
      {action}
    </div>
  );
}
function Modal({
  title,
  children,
  onClose,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  useEffect(() => {
    close.current = onClose;
  }, [onClose]);
  useEffect(() => {
    const prior = document.activeElement as HTMLElement | null;
    root.current?.querySelector<HTMLElement>("button,input,textarea")?.focus();
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") close.current();
      if (event.key === "Tab") {
        const items = root.current?.querySelectorAll<HTMLElement>(
          "button:not(:disabled),input,textarea,select,a[href]",
        );
        if (!items?.length) return;
        const first = items[0],
          last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("keydown", key);
      prior?.focus();
    };
  }, []);
  return (
    <div className="w-modal-backdrop">
      <div ref={root} role="dialog" aria-modal="true" aria-label={title} className="w-modal">
        <div className="w-modal-top">
          <h2>{title}</h2>
          <button className="w-icon-btn" aria-label="Close dialog" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

export default function WorldsApp() {
  const [area, setArea] = useState<Area>("home");
  const [projects, setProjects] = useState<WorldProject[]>([]);
  const [scenes, setScenes] = useState<WorldScene[]>([]);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [replays, setReplays] = useState<Replay[]>([]);
  const [assets, setAssets] = useState<MediaAsset[]>([]);
  const [reconstructions, setReconstructions] = useState<Reconstruction[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([]);
  const [settings, setSettingsState] = useState<AppSettings>(() => getSettings());
  const [toast, setToast] = useState("");
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All worlds");
  const [activeTemplate, setActiveTemplate] = useState<Template>(TEMPLATES[0]);
  const [draft, setDraft] = useState<WorldProject | null>(null);
  const [playing, setPlaying] = useState<{
    project: WorldProject;
    mode: "preview" | "live";
  } | null>(null);
  const [characterEditor, setCharacterEditor] = useState<Character | null>(null);
  const [replayViewer, setReplayViewer] = useState<Replay | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<{
    kind: "projects" | "characters" | "replays";
    id: string;
    name: string;
  } | null>(null);
  const [help, setHelp] = useState(false);
  const refresh = useCallback(async () => {
    try {
      const [p, s, c, r, a, x, b] = await Promise.all([
        worldStore.list("projects"),
        worldStore.list("scenes"),
        worldStore.list("characters"),
        worldStore.list("replays"),
        worldStore.list("assets"),
        worldStore.list("reconstructions"),
        worldStore.list("benchmarks"),
      ]);
      setProjects(p.sort((a, b) => b.updatedAt - a.updatedAt));
      setScenes(s);
      setCharacters(c);
      setReplays(r.sort((a, b) => b.createdAt - a.createdAt));
      setAssets(a);
      setReconstructions(x);
      setBenchmarks(b);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Your local library could not be opened.");
    }
  }, []);
  useEffect(() => {
    void Promise.resolve().then(refresh);
  }, [refresh]);
  useEffect(() => {
    if (!toast) return;
    const timeout = setTimeout(() => setToast(""), 4200);
    return () => clearTimeout(timeout);
  }, [toast]);
  const navigate = (next: Area) => {
    setArea(next);
    setQuery("");
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const newProject = (template?: Template): WorldProject => ({
    id: newId(),
    name: template?.name ?? "Untitled world",
    prompt: template?.prompt ?? "",
    modelId: MODELS.find((m) => m.status === "adapter-ready")?.id ?? "astronex-world",
    providerId: settings.defaultProvider,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    assetIds: [],
    characterIds: [],
    settings: { performance: "balanced" },
    templateId: template?.id,
  });
  const create = (template?: Template) => {
    setDraft(newProject(template));
    navigate("create");
  };
  const launch = async (project: WorldProject, mode: "preview" | "live") => {
    try {
      const saved = { ...project, updatedAt: Date.now(), previewOnly: mode === "preview" };
      await worldStore.put("projects", saved);
      await refresh();
      setPlaying({ project: saved, mode });
    } catch (e) {
      setError(String(e));
    }
  };
  const branch = (scene: WorldScene, now: number) => {
    const source = projects.find((p) => p.id === scene.projectId);
    if (!source) return;
    setDraft({
      ...source,
      id: newId(),
      name: `${source.name} · branch`,
      parentSceneId: scene.id,
      prompt: scene.prompt,
      assetIds: [
        ...new Set(
          scene.thumbnailAssetId ? [scene.thumbnailAssetId, ...scene.assetIds] : scene.assetIds,
        ),
      ],
      createdAt: now,
      updatedAt: now,
    });
    navigate("create");
    setToast("Branch uses a visual checkpoint; exact continuation is not guaranteed.");
  };
  const visibleTemplates = TEMPLATES.filter(
    (t) =>
      (category === "All worlds" ||
        (category === "Exploration" && ["redwood", "alien"].includes(t.id)) ||
        (category === "Adventure" && ["dungeon", "horror"].includes(t.id)) ||
        (category === "Dreamscapes" && t.id === "dream") ||
        (category === "From your media" && ["photo", "video"].includes(t.id))) &&
      (!query || `${t.name} ${t.tags.join(" ")}`.toLowerCase().includes(query.toLowerCase())),
  );
  return (
    <div className="worlds-app">
      <aside className="w-sidebar">
        <a className="w-brand" href="/worlds.html" aria-label="Worlds home">
          <span className="w-brand-mark">
            <Globe2 size={23} />
          </span>
          <span>
            worlds<span className="w-brand-dot">.</span>
          </span>
        </a>
        <div className="w-workspace">
          <span className="w-workspace-avatar">P</span>
          <span>
            Personal space<small>Local library</small>
          </span>
          <ChevronDown size={14} />
        </div>
        <button className="w-btn w-btn-primary w-create-nav" onClick={() => create()}>
          <Plus size={17} />
          Create a world
        </button>
        <div className="w-nav-label">YOUR UNIVERSE</div>
        <nav aria-label="Main navigation">
          {NAV.map((n) => (
            <button
              className={`w-nav-item ${area === n.id ? "active" : ""}`}
              aria-label={n.label}
              key={n.id}
              onClick={() => navigate(n.id)}
            >
              <n.icon size={18} />
              <span>{n.label}</span>
              {n.id === "worlds" && projects.length > 0 && <small>{projects.length}</small>}
              {n.id === "3d" && <span className="w-nav-new">LABS</span>}
            </button>
          ))}
        </nav>
        <div className="w-nav-label w-lab-label">WORKBENCH</div>
        <button
          className={`w-nav-item ${area === "lab" ? "active" : ""}`}
          onClick={() => navigate("lab")}
        >
          <FlaskConical size={18} />
          Model lab
        </button>
        <button
          className={`w-nav-item ${area === "settings" ? "active" : ""}`}
          onClick={() => navigate("settings")}
        >
          <Settings2 size={18} />
          Settings
        </button>
        <div className="w-sidebar-bottom">
          <div className="w-local-note">
            <span className="w-status-dot" />
            <span>
              Yours, by default.<small>Saved on this device.</small>
            </span>
            <ShieldCheck size={15} />
          </div>
          <a className="w-earth-link" href="/">
            <Globe2 size={15} />
            Open Hexapod Earth
            <ArrowRight size={14} />
          </a>
        </div>
      </aside>
      <div className="w-body">
        <header className="w-topbar">
          <div className="w-breadcrumb">
            Personal space
            <ChevronRight size={13} />
            <strong>
              {area === "home"
                ? "Discover"
                : area === "create"
                  ? "Create a world"
                  : area === "worlds"
                    ? "My worlds"
                    : area === "3d"
                      ? "3D worlds"
                      : area === "lab"
                        ? "Model lab"
                        : area.charAt(0).toUpperCase() + area.slice(1)}
            </strong>
          </div>
          <div className="w-top-actions">
            <span className="w-local-chip">
              <span className="w-status-dot" />
              LOCAL FIRST
            </span>
            <button className="w-icon-btn" aria-label="About Worlds" onClick={() => setHelp(true)}>
              <CircleHelp size={18} />
            </button>
            <span className="w-avatar">P</span>
          </div>
        </header>
        <main className={`w-main ${area === "home" ? "w-discover" : ""}`}>
          {error && (
            <div className="w-alert" role="alert">
              {error}
              <button
                className="w-icon-btn"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <X size={16} />
              </button>
            </div>
          )}
          {area === "home" && (
            <>
              <div className="w-page-heading">
                <div>
                  <div className="w-overline">IMAGINATION, OPEN WORLD.</div>
                  <h1>Where will you go?</h1>
                  <p>Dream it. Step inside. See what happens next.</p>
                </div>
                <div className="w-heading-note">
                  <span className="w-orbit" />
                  <span>
                    A new kind of
                    <br />
                    <strong>worldbuilding.</strong>
                  </span>
                </div>
              </div>
              <section className="w-hero" aria-label={`Featured world: ${activeTemplate.name}`}>
                <SceneArt kind={activeTemplate.kind} />
                <div className="w-hero-shade" />
                <div className="w-hero-content">
                  <span className="w-feature-badge">
                    <Sparkles size={12} />A WORLD TO GET LOST IN
                  </span>
                  <div className="w-hero-bottom">
                    <span className="w-hero-eyebrow">{activeTemplate.eyebrow}</span>
                    <h2>{activeTemplate.name}</h2>
                    <p>{activeTemplate.description}</p>
                    <div className="w-hero-buttons">
                      <button className="w-btn w-btn-light" onClick={() => create(activeTemplate)}>
                        Make it your world
                        <ArrowRight size={17} />
                      </button>
                      <button
                        className="w-preview-link"
                        onClick={() => void launch(newProject(activeTemplate), "preview")}
                      >
                        <Play size={14} />
                        Try interaction preview
                      </button>
                    </div>
                    <div className="w-hero-meta">
                      <span>
                        <Gamepad2 size={13} />
                        Model-powered exploration
                      </span>
                      <span>
                        <AudioLines size={13} />
                        Your words, new possibilities
                      </span>
                    </div>
                  </div>
                </div>
                <div className="w-hero-pagination">
                  {TEMPLATES.slice(0, 3).map((t, i) => (
                    <button
                      key={t.id}
                      aria-label={`Feature ${t.name}`}
                      onClick={() => setActiveTemplate(t)}
                      className={t.id === activeTemplate.id ? "active" : ""}
                    >
                      {String(i + 1).padStart(2, "0")}
                    </button>
                  ))}
                </div>
                <span className="w-art-label">CONCEPT ART · NOT MODEL OUTPUT</span>
              </section>
              <section className="w-imagine-bar">
                <div className="w-imagine-icon">
                  <WandSparkles size={22} />
                </div>
                <div>
                  <h3>Your imagination is a starting point.</h3>
                  <p>A sentence, a photo, a moment. Turn it into somewhere new.</p>
                </div>
                <button className="w-btn w-btn-quiet" onClick={() => create()}>
                  Create from scratch
                  <ArrowRight size={16} />
                </button>
              </section>
              <div className="w-section-heading">
                <div>
                  <h2>
                    Find your next world <span>08</span>
                  </h2>
                  <p>Start with a spark. Make the rest your own.</p>
                </div>
                <label className="w-search">
                  <Search size={15} />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Find a world…"
                    aria-label="Find a world"
                  />
                </label>
              </div>
              <div className="w-filter-row">
                {["All worlds", "Exploration", "Adventure", "Dreamscapes", "From your media"].map(
                  (c) => (
                    <button
                      key={c}
                      className={category === c ? "active" : ""}
                      onClick={() => setCategory(c)}
                    >
                      {c}
                    </button>
                  ),
                )}
                <span className="w-filter-caption">CURATED STARTING POINTS</span>
              </div>
              <div className="w-world-grid">
                {visibleTemplates.map((t) => (
                  <button className="w-world-card" key={t.id} onClick={() => create(t)}>
                    <div className="w-card-art">
                      <SceneArt kind={t.kind} />
                      <span className="w-card-tag">{t.tags[0]}</span>
                      <span className="w-card-enter">
                        <ArrowRight size={18} />
                      </span>
                      {["photo", "video"].includes(t.id) && (
                        <span className="w-input-art">
                          {t.id === "photo" ? <ImagePlus size={31} /> : <Video size={31} />}
                        </span>
                      )}
                    </div>
                    <div className="w-card-body">
                      <h3>
                        {t.name}
                        <ArrowRight size={15} />
                      </h3>
                      <p>{t.description}</p>
                      <span>
                        {t.tags[1]}
                        <i />
                        Template
                      </span>
                    </div>
                  </button>
                ))}
              </div>
              {visibleTemplates.length === 0 && (
                <p className="w-muted">No templates match your search.</p>
              )}
              <div className="w-discovery-footer">
                <span>
                  <ShieldCheck size={15} />
                  Your ideas live here. GPU inference only sends the inputs it needs.
                </span>
                <button onClick={() => navigate("settings")}>
                  Explore your setup
                  <ArrowRight size={14} />
                </button>
              </div>
            </>
          )}
          {area === "create" && draft && (
            <Composer
              project={draft}
              onChange={setDraft}
              characters={characters}
              settings={settings}
              onLaunch={launch}
              onError={setError}
              onSave={async () => {
                await worldStore.put("projects", { ...draft, updatedAt: Date.now() });
                await refresh();
                setToast("World saved to your local library.");
              }}
              onSettings={() => navigate("settings")}
              onCharacter={setCharacterEditor}
            />
          )}
          {area === "worlds" && (
            <>
              <PageHeading
                overline="PLACES YOU CAN COME BACK TO"
                title="My worlds"
                description="Your ideas, saved scenes, and roads not taken."
                action={
                  <button className="w-btn w-btn-primary" onClick={() => create()}>
                    <Plus size={16} />
                    New world
                  </button>
                }
              />
              {projects.length === 0 ? (
                <EmptyState
                  icon={FolderOpen}
                  title="Your next world starts here."
                  action={
                    <button className="w-btn w-btn-primary" onClick={() => create()}>
                      Create your first world
                      <ArrowRight size={16} />
                    </button>
                  }
                >
                  Create a world or try an interaction preview. Your projects and checkpoints stay
                  in this browser.
                </EmptyState>
              ) : (
                <div className="w-library-grid">
                  {projects.map((p) => (
                    <article className="w-saved-card" key={p.id}>
                      <div className="w-saved-art">
                        <SceneArt kind={p.templateId} />
                        {p.thumbnailAssetId && <AssetImage id={p.thumbnailAssetId} alt={p.name} />}
                        <span className="w-card-tag">
                          {p.previewOnly ? "Interaction preview" : "World project"}
                        </span>
                      </div>
                      <div className="w-card-body">
                        <h3>{p.name}</h3>
                        <p>{p.prompt}</p>
                        <div className="w-record-meta">
                          {getModel(p.modelId)?.name ?? p.modelId}
                          <span>{relativeTime(p.updatedAt)}</span>
                        </div>
                        <div className="w-card-actions">
                          <button
                            className="w-btn w-btn-quiet"
                            onClick={() => {
                              setDraft(p);
                              navigate("create");
                            }}
                          >
                            <SlidersHorizontal size={14} />
                            Open
                          </button>
                          <button
                            className="w-icon-btn"
                            title="Duplicate world"
                            aria-label={`Duplicate ${p.name}`}
                            onClick={() => {
                              void (async () => {
                                await worldStore.put("projects", {
                                  ...p,
                                  id: newId(),
                                  name: `${p.name} · copy`,
                                  createdAt: Date.now(),
                                  updatedAt: Date.now(),
                                });
                                await refresh();
                                setToast("World duplicated.");
                              })();
                            }}
                          >
                            <Copy size={15} />
                          </button>
                          <button
                            className="w-icon-btn"
                            aria-label={`Delete ${p.name}`}
                            onClick={() =>
                              setDeleteTarget({ kind: "projects", id: p.id, name: p.name })
                            }
                          >
                            <Trash2 size={15} />
                          </button>
                        </div>
                        {scenes
                          .filter((s) => s.projectId === p.id)
                          .map((s) => (
                            <button
                              className="w-checkpoint"
                              key={s.id}
                              onClick={() => branch(s, Date.now())}
                            >
                              <Layers3 size={14} />
                              <span>
                                {s.name}
                                <small>
                                  {s.resumeKind === "exact"
                                    ? "Exact resume"
                                    : s.resumeKind === "approximate"
                                      ? "Approximate resume"
                                      : "Visual checkpoint"}
                                </small>
                              </span>
                              <span>
                                Branch
                                <ArrowRight size={13} />
                              </span>
                            </button>
                          ))}
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </>
          )}
          {area === "characters" && (
            <>
              <PageHeading
                overline="FAMILIAR FACES, UNFAMILIAR PLACES"
                title="Characters"
                description="Keep a reference. Take it somewhere new."
                action={
                  <button
                    className="w-btn w-btn-primary"
                    onClick={() =>
                      setCharacterEditor({
                        id: newId(),
                        name: "",
                        description: "",
                        assetIds: [],
                        createdAt: Date.now(),
                        updatedAt: Date.now(),
                        identityMethod: "reference-images",
                      })
                    }
                  >
                    <Plus size={16} />
                    Create character
                  </button>
                }
              />
              <div className="w-info-strip">
                <ShieldCheck size={17} />
                <span>
                  References are stored on this device. Identity consistency depends on the model
                  and is never guaranteed.
                </span>
              </div>
              {characters.length === 0 ? (
                <EmptyState
                  icon={UsersRound}
                  title="A familiar face in any world."
                  action={
                    <button
                      className="w-btn w-btn-quiet"
                      onClick={() =>
                        setCharacterEditor({
                          id: newId(),
                          name: "",
                          description: "",
                          assetIds: [],
                          createdAt: Date.now(),
                          updatedAt: Date.now(),
                        })
                      }
                    >
                      <Plus size={16} />
                      Create a character
                    </button>
                  }
                >
                  Save a description and reference images or video. Models with character
                  conditioning can use the same package across scenes.
                </EmptyState>
              ) : (
                <div className="w-library-grid">
                  {characters.map((c) => (
                    <article className="w-character-card" key={c.id}>
                      <div className="w-character-portrait">
                        <UsersRound size={46} />
                        <AssetImage
                          id={c.assetIds.find(
                            (id) => assets.find((a) => a.id === id)?.kind === "image",
                          )}
                          alt={c.name}
                        />
                      </div>
                      <div className="w-card-body">
                        <h3>{c.name}</h3>
                        <p>{c.description}</p>
                        <div className="w-record-meta">
                          {c.assetIds.length} references
                          <span>{c.identityMethod ?? "Unverified"}</span>
                        </div>
                        <div className="w-card-actions">
                          <button
                            className="w-btn w-btn-quiet"
                            onClick={() => {
                              setDraft({ ...newProject(), characterIds: [c.id] });
                              navigate("create");
                            }}
                          >
                            Add to world
                            <ArrowRight size={14} />
                          </button>
                          <button
                            className="w-icon-btn"
                            aria-label={`Edit ${c.name}`}
                            onClick={() => setCharacterEditor(c)}
                          >
                            <SlidersHorizontal size={15} />
                          </button>
                          <button
                            className="w-icon-btn"
                            aria-label={`Delete ${c.name}`}
                            onClick={() =>
                              setDeleteTarget({ kind: "characters", id: c.id, name: c.name })
                            }
                          >
                            <Trash2 size={15} />
                          </button>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </>
          )}
          {area === "replays" && (
            <>
              <PageHeading
                overline="SOME MOMENTS DESERVE ANOTHER LOOK"
                title="Replays"
                description="The places you went. Everything that happened."
              />
              {replays.length === 0 ? (
                <EmptyState
                  icon={Video}
                  title="Take the moment with you."
                  action={
                    <button className="w-btn w-btn-primary" onClick={() => create()}>
                      Explore a world
                      <ArrowRight size={16} />
                    </button>
                  }
                >
                  Start recording from the player. Your video and timestamped control history will
                  be saved here for playback and download.
                </EmptyState>
              ) : (
                <div className="w-library-grid">
                  {replays.map((r) => (
                    <article className="w-saved-card" key={r.id}>
                      <button className="w-replay-art" onClick={() => setReplayViewer(r)}>
                        <SceneArt kind={projects.find((p) => p.id === r.projectId)?.templateId} />
                        <span className="w-play-circle">
                          <Play size={24} />
                        </span>
                        <span className="w-card-tag">
                          {r.previewOnly ? "Preview recording" : "Recorded session"}
                        </span>
                      </button>
                      <div className="w-card-body">
                        <h3>{r.name}</h3>
                        <div className="w-record-meta">
                          {Math.round(r.durationMs / 1000)} seconds
                          <span>{relativeTime(r.createdAt)}</span>
                        </div>
                        <div className="w-card-actions">
                          <button
                            className="w-btn w-btn-quiet"
                            onClick={() => {
                              void (async () => {
                                const blob = await worldStore.getBlob(r.assetId);
                                if (blob) downloadBlob(blob, `${r.name}.webm`);
                                else setError("The replay file is missing from local storage.");
                              })();
                            }}
                          >
                            <Download size={14} />
                            Download
                          </button>
                          <button className="w-btn w-btn-quiet" onClick={() => navigate("3d")}>
                            <Box size={14} />
                            To 3D
                          </button>
                          <button
                            className="w-icon-btn"
                            aria-label={`Delete ${r.name}`}
                            onClick={() =>
                              setDeleteTarget({ kind: "replays", id: r.id, name: r.name })
                            }
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </>
          )}
          {area === "3d" && (
            <>
              <PageHeading
                overline="MAKE A PLACE YOU CAN KEEP"
                title="3D worlds"
                description="Reconstruct an explored world from the frames you captured."
              />
              <ReconstructionPanel
                projects={projects}
                replays={replays}
                assets={assets}
                reconstructions={reconstructions}
                getAssetBlob={(id) => worldStore.getBlob(id)}
                onSave={async (record) => {
                  await worldStore.put("reconstructions", record);
                  await refresh();
                }}
                apiBaseUrl={settings.apiBaseUrl}
              />
            </>
          )}
          {area === "lab" && (
            <ModelLab
              projects={projects}
              benchmarks={benchmarks}
              characters={characters}
              serverUrl={settings.apiBaseUrl}
              onSaved={() => void refresh()}
              onCreate={(modelId) => {
                setDraft({ ...newProject(), modelId });
                navigate("create");
              }}
            />
          )}
          {area === "settings" && (
            <SettingsView
              settings={settings}
              onSave={(next) => {
                saveSettings(next);
                setSettingsState(next);
                setToast("Settings saved on this device.");
              }}
              onRefresh={refresh}
              onError={setError}
              onToast={setToast}
            />
          )}
        </main>
        <footer className="w-footer">
          <span>
            WORLDS <i /> A NEW SPACE FOR YOUR IMAGINATION
          </span>
          <span>Experimental models. Real possibilities.</span>
        </footer>
      </div>
      {toast && (
        <div className="w-toast" role="status">
          <CheckCircle2 size={18} />
          {toast}
        </div>
      )}
      {playing && (
        <Player
          project={playing.project}
          mode={playing.mode}
          serverUrl={settings.apiBaseUrl}
          onExit={() => {
            setPlaying(null);
            void refresh();
          }}
          onSaved={() => void refresh()}
        />
      )}
      {characterEditor && (
        <CharacterEditor
          character={characterEditor}
          serverUrl={settings.apiBaseUrl}
          initialScene={draft?.prompt}
          onUseReference={(assetId, prompt) => {
            setDraft({
              ...(draft ?? newProject()),
              prompt,
              assetIds: [assetId],
              characterIds: [],
              updatedAt: Date.now(),
            });
            setCharacterEditor(null);
            navigate("create");
          }}
          onClose={() => setCharacterEditor(null)}
          onSave={async (c) => {
            await worldStore.put("characters", c);
            await refresh();
            setCharacterEditor(null);
            setToast("Character saved to your local library.");
          }}
        />
      )}
      {replayViewer && (
        <ReplayViewer
          replay={replayViewer}
          serverUrl={settings.apiBaseUrl}
          onClose={() => setReplayViewer(null)}
        />
      )}
      {deleteTarget && (
        <Modal title={`Delete ${deleteTarget.name}?`} onClose={() => setDeleteTarget(null)}>
          <p className="w-muted">
            This removes the{" "}
            {deleteTarget.kind === "projects"
              ? "project"
              : deleteTarget.kind === "characters"
                ? "character"
                : "replay"}{" "}
            from this browser. This action cannot be undone.
          </p>
          <div className="w-modal-actions">
            <button className="w-btn w-btn-quiet" onClick={() => setDeleteTarget(null)}>
              Keep it
            </button>
            <button
              className="w-btn w-btn-danger"
              onClick={() => {
                void (async () => {
                  await worldStore.remove(deleteTarget.kind, deleteTarget.id);
                  setDeleteTarget(null);
                  await refresh();
                  setToast("Removed from your library.");
                })();
              }}
            >
              Delete
            </button>
          </div>
        </Modal>
      )}
      {help && (
        <Modal title="Welcome to Worlds." onClose={() => setHelp(false)}>
          <p className="w-modal-lead">A place to explore what neural world models can become.</p>
          <p className="w-muted">
            Create a prompt, choose a model, and connect a GPU worker. RunPod is the default compute
            provider; your world library stays on this device.
          </p>
          <div className="w-info-strip">
            <Monitor size={20} />
            <span>
              Interaction previews are procedural interface demonstrations, not AI-generated worlds.
              Real generation requires a compatible worker.
            </span>
          </div>
          <p className="w-muted">
            Worlds and Hexapod Earth are separate products accessible on the same domain.
          </p>
          <button
            className="w-btn w-btn-primary"
            onClick={() => {
              setHelp(false);
              navigate("settings");
            }}
          >
            Configure compute
            <ArrowRight size={16} />
          </button>
        </Modal>
      )}
    </div>
  );
}

function PageHeading({
  overline,
  title,
  description,
  action,
}: {
  overline: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="w-page-heading">
      <div>
        <div className="w-overline">{overline}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}

function Composer({
  project,
  onChange,
  characters,
  settings,
  onLaunch,
  onError,
  onSave,
  onSettings,
  onCharacter,
}: {
  project: WorldProject;
  onChange: (p: WorldProject) => void;
  characters: Character[];
  settings: AppSettings;
  onLaunch: (p: WorldProject, mode: "preview" | "live") => Promise<void>;
  onError: (message: string) => void;
  onSave: () => Promise<void>;
  onSettings: () => void;
  onCharacter: (character: Character) => void;
}) {
  const model = getModel(project.modelId);
  const caps = model.capabilities;
  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [busy, setBusy] = useState("");
  const [enhancement, setEnhancement] = useState<{
    original: string;
    enhanced: string;
    source: "llm" | "local-guide";
    notes?: string[];
  } | null>(null);
  const [game, setGame] = useState<Awaited<
    ReturnType<ReturnType<typeof createWorldApi>["createGame"]>
  > | null>(null);
  const readiness = useComputeReadiness(settings.apiBaseUrl);
  const [consent, setConsent] = useState(false);
  const change = (patch: Partial<WorldProject>) => onChange({ ...project, ...patch });
  useEffect(() => {
    void worldStore
      .list("assets")
      .then(setMedia)
      .catch((e: unknown) => onError(String(e)));
  }, [project.assetIds, onError]);
  const provider = readiness.report?.providers.find((p) => p.id === project.providerId);
  const compute = readiness.report
    ? assessCompute(readiness.report, project.providerId, project.modelId)
    : undefined;
  const remote = project.providerId !== "local";
  const modelAssets = media.filter((a) => project.assetIds.includes(a.id));
  const inputIssue =
    project.settings.seed !== undefined &&
    (!Number.isInteger(project.settings.seed) ||
      project.settings.seed < 0 ||
      project.settings.seed > 4294967295)
      ? "Seed must be a whole number from 0 to 4294967295."
      : project.assetIds.length !== modelAssets.length
        ? "Some reference media is still loading or missing from this device. Remove unavailable references before starting compute."
        : modelAssets.reduce((total, asset) => total + asset.size, 0) > 4 * 1024 * 1024
          ? "References exceed the 4 MB session limit. Resize or shorten them before starting a GPU worker."
          : modelAssets.some(
                (a) =>
                  a.kind === "image" &&
                  !["image/jpeg", "image/png", "image/webp"].includes(a.mimeType),
              )
            ? "Live image inputs must be JPEG, PNG, or WebP. Convert this reference before starting a worker."
            : modelAssets.some((a) => a.kind === "video") && !caps.input.video
              ? "This model does not support video conditioning. Choose another model or remove the video."
              : modelAssets.filter((a) => a.kind === "image").length > 1 && !caps.input.multiImage
                ? "This model accepts one image. Remove extra images or select a model with multi-image support."
                : modelAssets.some((a) => a.kind === "image") && !caps.input.image
                  ? "This model does not support image conditioning."
                  : project.characterIds.length && !caps.control.characterReference
                    ? "Prepare a character scene below to use image conditioning with this model, or remove the character."
                    : caps.input.requiredImage && !modelAssets.some((a) => a.kind === "image")
                      ? "This model starts from an image. Add a reference image before starting compute."
                      : "";
  const upload = async (files: File[]) => {
    try {
      setBusy("upload");
      const ids: string[] = [];
      for (const file of files) {
        if (!file.type.startsWith("image/") && !file.type.startsWith("video/"))
          throw new Error("Add an image or video file.");
        if (file.size > 150 * 1024 * 1024) throw new Error("Each reference must be under 150 MB.");
        const asset = await worldStore.saveAsset(file, file.name);
        ids.push(asset.id);
      }
      change({ assetIds: [...project.assetIds, ...ids] });
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };
  const enhance = async () => {
    if (!project.prompt.trim()) return;
    setBusy("enhance");
    try {
      setEnhancement(
        await createWorldApi(settings.apiBaseUrl).enhancePrompt({
          prompt: project.prompt,
          modelId: model.id,
          capabilities: caps,
          experience: "exploration",
        }),
      );
    } catch {
      setEnhancement({
        original: project.prompt,
        enhanced: `${project.prompt.trim().replace(/[.!?]$/, "")}. ${caps.control.wasd ? "First-person viewpoint with navigable paths, clear foreground detail, and a recognizable distant landmark." : "Establish a clear viewpoint, coherent scene layout, and a recognizable distant landmark."} Consistent lighting, natural scale, rich environmental detail, and subtle atmospheric motion.`,
        source: "local-guide",
        notes: [
          "Your AI service could not be reached. This is an editable local prompt-writing guide, not an LLM result.",
        ],
      });
    } finally {
      setBusy("");
    }
  };
  const makeGame = async () => {
    setBusy("game");
    try {
      setGame(
        await createWorldApi(settings.apiBaseUrl).createGame({
          prompt: project.prompt,
          modelId: model.id,
          capabilities: caps,
          nativeActions: model.nativeActions,
        }),
      );
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };
  const canLive =
    (!caps.input.text || project.prompt.trim().length > 0) &&
    model.status === "adapter-ready" &&
    compute?.canLaunch &&
    !readiness.checking &&
    !inputIssue &&
    (!remote || consent);
  return (
    <>
      <PageHeading
        overline="EVERY WORLD STARTS WITH AN IDEA"
        title="Imagine somewhere new."
        description="Give it a beginning. The model takes it from there."
        action={
          <button
            className="w-btn w-btn-quiet"
            disabled={!!busy || !project.prompt.trim()}
            onClick={() => {
              setBusy("save");
              void onSave()
                .catch((e: unknown) => onError(String(e)))
                .finally(() => setBusy(""));
            }}
          >
            <FolderOpen size={15} />
            Save draft
          </button>
        }
      />
      <div className="w-compose-layout">
        <div>
          <section className="w-panel">
            <label className="w-form-field">
              <span>WORLD NAME</span>
              <input
                value={project.name}
                maxLength={100}
                onChange={(e) => change({ name: e.target.value })}
                placeholder="Give this place a name"
              />
            </label>
            <div
              className="w-composer-prompt"
              onDrop={(e) => {
                e.preventDefault();
                void upload(Array.from(e.dataTransfer.files));
              }}
              onDragOver={(e) => e.preventDefault()}
            >
              <textarea
                aria-label="World prompt"
                maxLength={16000}
                placeholder="A huge abandoned industrial complex in a stormy tropical jungle…"
                value={project.prompt}
                onChange={(e) => change({ prompt: e.target.value })}
                onPaste={(e) => {
                  const files = Array.from(e.clipboardData.files);
                  if (files.length) {
                    e.preventDefault();
                    void upload(files);
                  }
                }}
              />
              <div className="w-composer-prompt-footer">
                <span>DESCRIBE YOUR WORLD</span>
                <div className="w-compose-tools">
                  <button
                    className="w-btn w-btn-quiet"
                    disabled={!!busy || !project.prompt.trim()}
                    onClick={() => void makeGame()}
                  >
                    {busy === "game" ? (
                      <Loader2 size={13} className="w-spin" />
                    ) : (
                      <Gamepad2 size={13} />
                    )}
                    Create a game
                  </button>
                  <button
                    className="w-btn w-btn-quiet"
                    disabled={!!busy || !project.prompt.trim()}
                    onClick={() => void enhance()}
                  >
                    {busy === "enhance" ? (
                      <Loader2 size={13} className="w-spin" />
                    ) : (
                      <Sparkles size={13} />
                    )}
                    Enhance prompt
                  </button>
                </div>
              </div>
            </div>
            <div className="w-composer-upload">
              <label className="w-upload">
                <ImagePlus size={19} />
                <span>
                  Add images or a video <small>· or drop / paste here</small>
                </span>
                <input
                  className="w-hidden-input"
                  type="file"
                  accept="image/*,video/*"
                  multiple
                  onChange={(e) => {
                    void upload(Array.from(e.target.files ?? []));
                    e.target.value = "";
                  }}
                />
              </label>
              <div className="w-upload-grid">
                {modelAssets.map((a) => (
                  <div className="w-upload-file" key={a.id}>
                    {a.kind === "image" ? (
                      <AssetImage id={a.id} alt={a.name} />
                    ) : (
                      <Video size={19} />
                    )}
                    <span>{a.name}</span>
                    <button
                      className="w-icon-btn"
                      aria-label={`Remove ${a.name}`}
                      onClick={() =>
                        change({ assetIds: project.assetIds.filter((id) => id !== a.id) })
                      }
                    >
                      <X size={13} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
            {inputIssue && <p className="w-inline-error">{inputIssue}</p>}
            {caps.input.image && (
              <MediaConditioning
                project={project}
                assets={modelAssets}
                serverUrl={settings.apiBaseUrl}
                onChange={onChange}
              />
            )}
          </section>
          <section className="w-panel">
            <h2>A familiar face, if you like.</h2>
            <p className="w-panel-subtitle">
              Add a character from your library. Compatibility depends on the model’s conditioning
              interface.
            </p>
            {characters.length > 0 ? (
              <div className="w-character-chips">
                {characters.map((c) => (
                  <button
                    key={c.id}
                    className={project.characterIds.includes(c.id) ? "active" : ""}
                    onClick={() =>
                      change({
                        characterIds: project.characterIds.includes(c.id)
                          ? project.characterIds.filter((id) => id !== c.id)
                          : [...project.characterIds, c.id],
                      })
                    }
                  >
                    {project.characterIds.includes(c.id) ? <Check size={13} /> : <Plus size={13} />}{" "}
                    {c.name}
                  </button>
                ))}
              </div>
            ) : (
              <p className="w-muted">
                Create reusable references in Characters, then bring them here.
              </p>
            )}
            {characters
              .filter((character) => project.characterIds.includes(character.id))
              .map((character) => (
                <button
                  key={character.id}
                  className="w-btn w-btn-quiet"
                  onClick={() => onCharacter(character)}
                >
                  <Sparkles size={14} /> Prepare a scene with {character.name}
                </button>
              ))}
          </section>
          <div className="w-compute-note">
            <ShieldCheck size={15} />
            <span>
              Your draft and reference media stay on this device. Starting a remote session sends
              its prompt and required input media to your selected GPU worker. Do not include
              information you do not want that worker to process.
            </span>
          </div>
        </div>
        <aside className="w-compose-settings">
          <section className="w-panel">
            <div className="w-compose-preview">
              <SceneArt kind={project.templateId} />
            </div>
            <label className="w-form-field">
              <span>WORLD MODEL</span>
              <select
                value={project.modelId}
                onChange={(e) =>
                  change({
                    modelId: e.target.value,
                    settings: {
                      ...project.settings,
                      performance: "balanced",
                      resolution: undefined,
                    },
                  })
                }
              >
                {MODELS.map((m) => (
                  <option value={m.id} key={m.id}>
                    {m.name}
                    {m.status === "research" ? " · research" : ""}
                  </option>
                ))}
              </select>
            </label>
            <p className="w-select-description">{model.description}</p>
            <label className="w-form-field">
              <span>COMPUTE PROVIDER</span>
              <select
                value={project.providerId}
                onChange={(e) => {
                  change({ providerId: e.target.value as ProviderId });
                  setConsent(false);
                }}
              >
                {PROVIDERS.map((p) => (
                  <option value={p.id} key={p.id}>
                    {p.name}
                    {p.id === "runpod" ? " · default" : ""}
                  </option>
                ))}
              </select>
            </label>
            <span className="w-field-label">PERFORMANCE</span>
            <div className="w-quality-picker">
              {(["quality", "balanced", "low-latency"] as const).map((v) => (
                <button
                  key={v}
                  className={project.settings.performance === v ? "active" : ""}
                  disabled={
                    caps.runtime.qualityOptions !== undefined &&
                    !caps.runtime.qualityOptions.includes(v)
                  }
                  onClick={() => change({ settings: { ...project.settings, performance: v } })}
                >
                  {v === "quality" ? "Quality" : v === "balanced" ? "Balanced" : "Low latency"}
                </button>
              ))}
            </div>
            <details>
              <summary>Advanced settings</summary>
              <label className="w-form-field">
                <span>
                  Seed {caps.persistence.deterministicSeed ? "" : "· not verified for this model"}
                </span>
                <input
                  type="number"
                  min={0}
                  max={4294967295}
                  step={1}
                  disabled={!caps.persistence.deterministicSeed}
                  value={project.settings.seed ?? ""}
                  placeholder="Random"
                  onChange={(e) =>
                    change({
                      settings: {
                        ...project.settings,
                        seed: e.target.value ? Number(e.target.value) : undefined,
                      },
                    })
                  }
                />
              </label>
              {caps.runtime.resolutionOptions.length > 0 && (
                <label className="w-form-field">
                  <span>Resolution</span>
                  <select
                    value={project.settings.resolution ?? ""}
                    onChange={(e) =>
                      change({
                        settings: { ...project.settings, resolution: e.target.value || undefined },
                      })
                    }
                  >
                    <option value="">Model default</option>
                    {caps.runtime.resolutionOptions.map((r) => (
                      <option key={r}>{r}</option>
                    ))}
                  </select>
                </label>
              )}
              <p className="w-select-description">
                Inference steps, quantization, attention, and checkpoint selection are managed by
                the configured worker. Unsupported overrides are not sent.
              </p>
            </details>
            <div className="w-model-notes">{model.caveat}</div>
          </section>
          <section className="w-panel w-launch-panel">
            <h2>Ready when you are.</h2>
            {readiness.checking && <p role="status">Checking worker and model readiness…</p>}
            {readiness.error && (
              <p className="w-inline-error" role="alert">
                {readiness.error}
              </p>
            )}
            {readiness.report && (
              <ReadinessDetails
                report={readiness.report}
                providerId={project.providerId}
                modelId={project.modelId}
                compact
              />
            )}
            <ReadinessRefresh checking={readiness.checking} onRefresh={readiness.refresh} />
            {remote && provider?.configured && (
              <label className="w-check-label">
                <input
                  type="checkbox"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                />
                Send required inputs to the remote worker.{" "}
                {compute?.kind === "provision"
                  ? "Allocate a paid GPU using the approved provider template."
                  : "Existing compute may continue incurring provider charges."}
              </label>
            )}
            <button
              className="w-btn w-btn-primary"
              disabled={!canLive || !!busy}
              onClick={() => void onLaunch(project, "live")}
            >
              <Play size={15} />
              {compute?.kind === "provision" ? "Start paid GPU world" : "Start GPU world"}
            </button>
            {!compute?.canLaunch && (
              <button className="w-btn w-btn-quiet" onClick={onSettings}>
                <Settings2 size={14} />
                Configure compute
              </button>
            )}
            <div className="w-preview-divider">Explore the interface</div>
            <button
              className="w-btn w-btn-quiet"
              disabled={!project.prompt.trim() || !!busy}
              onClick={() => void onLaunch(project, "preview")}
            >
              <Monitor size={15} />
              Try interaction preview
            </button>
            <p>
              Preview is a procedural control demonstration. It does not run the selected AI model.
            </p>
            <p>
              GPU pricing and performance are unmeasured until a configured provider supplies them.
            </p>
          </section>
        </aside>
      </div>
      {enhancement && (
        <Modal title="A little more imagination." onClose={() => setEnhancement(null)}>
          <span className="w-source-badge">
            {enhancement.source === "llm" ? "AI-ENHANCED PROMPT" : "LOCAL WRITING GUIDE · NOT AI"}
          </span>
          <span className="w-field-label">ORIGINAL</span>
          <p className="w-enhance-original">{enhancement.original}</p>
          <label className="w-form-field">
            <span>ENHANCED · EDIT BEFORE APPLYING</span>
            <textarea
              style={{ minHeight: 180 }}
              value={enhancement.enhanced}
              onChange={(e) => setEnhancement({ ...enhancement, enhanced: e.target.value })}
            />
          </label>
          {enhancement.notes?.map((note, i) => (
            <p className="w-muted" key={i}>
              {note}
            </p>
          ))}
          <div className="w-modal-actions">
            <button className="w-btn w-btn-quiet" onClick={() => setEnhancement(null)}>
              Keep original
            </button>
            <button className="w-btn w-btn-quiet" disabled={!!busy} onClick={() => void enhance()}>
              <Sparkles size={14} />
              Regenerate
            </button>
            <button
              className="w-btn w-btn-primary"
              onClick={() => {
                change({ prompt: enhancement.enhanced });
                setEnhancement(null);
              }}
            >
              Use this prompt
              <Check size={15} />
            </button>
          </div>
        </Modal>
      )}
      {game && (
        <Modal title={game.name} onClose={() => setGame(null)}>
          <span className="w-source-badge">
            {game.source === "llm" ? "AI GAME CONCEPT" : "LOCAL GAME GUIDE · NOT AI"}
          </span>
          <p className="w-modal-lead">{game.premise}</p>
          <p className="w-muted">
            <strong>Suggested objective:</strong> {game.objective}
          </p>
          <label className="w-form-field">
            <span>STARTING SCENE</span>
            <textarea
              value={game.prompt}
              onChange={(e) => setGame({ ...game, prompt: e.target.value })}
            />
          </label>
          {game.events.length > 0 && (
            <div className="w-code-block">
              {game.events.map((e) => `${e.atSeconds}s · ${e.prompt}`).join("\n")}
            </div>
          )}
          <p className="w-muted">
            The objective and event schedule stay with this world. In the player, enable the game
            director to apply events and assess progress from generated frames. Assessments are
            model judgments, not authoritative game rules.
          </p>
          <div className="w-modal-actions">
            <button className="w-btn w-btn-quiet" onClick={() => setGame(null)}>
              Keep original
            </button>
            <button
              className="w-btn w-btn-primary"
              onClick={() => {
                change({
                  name: game.name,
                  prompt: game.prompt,
                  game: { objective: game.objective, events: game.events },
                });
                setGame(null);
              }}
            >
              Use this concept
              <ArrowRight size={15} />
            </button>
          </div>
        </Modal>
      )}
    </>
  );
}

function CharacterEditor({
  character,
  onClose,
  onSave,
  serverUrl,
  onUseReference,
  initialScene,
}: {
  character: Character;
  serverUrl: string;
  initialScene?: string;
  onUseReference: (assetId: string, prompt: string) => void;
  onClose: () => void;
  onSave: (c: Character) => Promise<void>;
}) {
  const [draft, setDraft] = useState(character);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [assets, setAssets] = useState<MediaAsset[]>([]);
  useEffect(() => {
    void worldStore.list("assets").then(setAssets);
  }, [draft.assetIds.length]);
  return (
    <Modal title={character.name ? "Edit character" : "Meet someone new."} onClose={onClose}>
      <label className="w-form-field">
        <span>NAME</span>
        <input
          value={draft.name}
          maxLength={100}
          placeholder="What should we call them?"
          onChange={(e) => setDraft({ ...draft, name: e.target.value })}
        />
      </label>
      <label className="w-form-field">
        <span>DESCRIPTION</span>
        <textarea
          value={draft.description}
          placeholder="Appearance, clothing, distinguishing details, and a little personality…"
          onChange={(e) => setDraft({ ...draft, description: e.target.value })}
        />
      </label>
      <label className="w-upload">
        <ImagePlus size={20} />
        Add reference images or video
        <input
          type="file"
          className="w-hidden-input"
          accept="image/*,video/*"
          multiple
          onChange={(e) => {
            void (async () => {
              const files = Array.from(e.target.files ?? []);
              e.target.value = "";
              setBusy(true);
              setError("");
              try {
                const ids: string[] = [];
                for (const file of files) {
                  if (file.size > 150 * 1024 * 1024)
                    throw new Error("Each reference must be under 150 MB.");
                  if (!/^(image|video)\//.test(file.type))
                    throw new Error("Only image and video references are supported.");
                  const a = await worldStore.saveAsset(file, file.name);
                  ids.push(a.id);
                }
                setDraft((p) => ({ ...p, assetIds: [...p.assetIds, ...ids] }));
              } catch (e) {
                setError(String(e));
              } finally {
                setBusy(false);
              }
            })();
          }}
        />
      </label>
      <div className="w-upload-grid">
        {draft.assetIds.map((id) => (
          <div className="w-upload-file" key={id}>
            <AssetImage
              id={assets.find((a) => a.id === id)?.kind === "image" ? id : undefined}
              alt="Character reference"
            />
            <span>{assets.find((a) => a.id === id)?.name ?? "Reference"}</span>
            <button
              className="w-icon-btn"
              aria-label="Remove reference"
              onClick={() =>
                setDraft({ ...draft, assetIds: draft.assetIds.filter((a) => a !== id) })
              }
            >
              <X size={13} />
            </button>
          </div>
        ))}
      </div>
      <p className="w-muted" style={{ marginTop: 18 }}>
        A reference package is saved locally. Portrait synthesis, identity training, and consistency
        evaluation require a compatible model service.
      </p>
      <CharacterStudio
        serverUrl={serverUrl}
        character={draft}
        initialScene={initialScene}
        onUseReference={onUseReference}
        onSave={async (next) => {
          await worldStore.put("characters", next);
          setDraft(next);
        }}
      />
      {error && (
        <p className="w-inline-error" role="alert">
          {error}
        </p>
      )}
      <div className="w-modal-actions">
        <button className="w-btn w-btn-quiet" onClick={onClose}>
          Cancel
        </button>
        <button
          className="w-btn w-btn-primary"
          disabled={!draft.name.trim() || busy}
          onClick={() => {
            void (async () => {
              setBusy(true);
              try {
                await onSave({
                  ...draft,
                  name: draft.name.trim(),
                  updatedAt: Date.now(),
                  identityMethod: draft.assetIds.length ? "reference-images" : "prompt",
                });
              } catch (e) {
                setError(String(e));
              } finally {
                setBusy(false);
              }
            })();
          }}
        >
          {busy ? <Loader2 size={15} className="w-spin" /> : <Check size={15} />}Save character
        </button>
      </div>
    </Modal>
  );
}
function ReplayViewer({
  replay,
  onClose,
  serverUrl,
}: {
  replay: Replay;
  onClose: () => void;
  serverUrl: string;
}) {
  const [url, setUrl] = useState("");
  const [videoBlob, setVideoBlob] = useState<Blob>();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let objectUrl = "";
    let active = true;
    void worldStore
      .getBlob(replay.assetId)
      .then((blob) => {
        if (!active) return;
        if (blob) {
          objectUrl = URL.createObjectURL(blob);
          setUrl(objectUrl);
          setVideoBlob(blob);
        } else setError("This replay file is missing from your local library.");
      })
      .catch((e: unknown) => setError(String(e)));
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [replay.assetId]);
  return (
    <Modal title={replay.name} onClose={onClose}>
      {url ? (
        <video ref={videoRef} src={url} controls autoPlay playsInline>
          <track kind="captions" label="No captions available" />
        </video>
      ) : (
        <p className="w-muted">{error || "Opening your recording…"}</p>
      )}
      <p className="w-muted" style={{ marginTop: 15 }}>
        {replay.previewOnly
          ? "Interaction preview recording · not AI output."
          : "Recorded model session."}{" "}
        {replay.events.length} control events captured.
      </p>
      {videoBlob && !replay.previewOnly && (
        <ReplayHighlights
          serverUrl={serverUrl}
          video={videoBlob}
          durationSeconds={replay.durationMs / 1000}
          onSeek={(seconds) => {
            if (videoRef.current) videoRef.current.currentTime = seconds;
          }}
        />
      )}
      <div className="w-modal-actions">
        <button
          className="w-btn w-btn-quiet"
          onClick={() =>
            downloadBlob(
              new Blob(
                [
                  JSON.stringify(
                    {
                      version: 1,
                      modelId: replay.modelId,
                      durationMs: replay.durationMs,
                      events: replay.events,
                    },
                    null,
                    2,
                  ),
                ],
                { type: "application/json" },
              ),
              `${replay.name}-controls.json`,
            )
          }
        >
          <Download size={15} />
          Control trajectory
        </button>
        {url && (
          <a className="w-btn w-btn-primary" href={url} download={`${replay.name}.webm`}>
            <Download size={15} />
            Download video
          </a>
        )}
      </div>
    </Modal>
  );
}

function ModelLab({
  projects,
  benchmarks,
  characters,
  onCreate,
  serverUrl,
  onSaved,
}: {
  projects: WorldProject[];
  benchmarks: BenchmarkResult[];
  characters: Character[];
  serverUrl: string;
  onSaved: () => void;
  onCreate: (modelId: string) => void;
}) {
  const [selected, setSelected] = useState<string[]>(MODELS.slice(0, 2).map((m) => m.id));
  const [projectId, setProjectId] = useState(projects[0]?.id ?? "");
  const [showRaw, setShowRaw] = useState(false);
  const [labTab, setLabTab] = useState<"models" | "compare" | "experiments" | "customization">(
    "models",
  );
  const comparison = MODELS.filter((m) => selected.includes(m.id));
  const criteria: [string, (m: (typeof MODELS)[number]) => boolean][] = [
    ["Text input", (m) => m.capabilities.input.text],
    ["Image input", (m) => m.capabilities.input.image],
    ["Video input", (m) => m.capabilities.input.video],
    ["Movement controls", (m) => m.capabilities.control.wasd],
    ["Prompt-adapted movement", (m) => m.capabilities.runtime.controlMode === "prompt-adapted"],
    ["Camera control", (m) => m.capabilities.control.mouseLook],
    ["Live prompt updates", (m) => m.capabilities.control.promptDuringRollout],
    ["Character references", (m) => m.capabilities.control.characterReference],
    ["Native audio", (m) => m.capabilities.output.audio],
    ["Snapshot / restore", (m) => m.capabilities.persistence.snapshotRestore],
    ["Camera poses", (m) => m.capabilities.output.cameraPose],
  ];
  return (
    <>
      <PageHeading
        overline="CURIOSITY, WITH RECEIPTS"
        title="Model lab"
        description="Different models. Different possibilities. Explore the evidence."
      />
      <div className="w-info-strip">
        <FlaskConical size={18} />
        <span>
          Adapter capabilities describe this integration. Research candidates are not launchable. No
          FPS, cost, identity, or consistency result is claimed without a measured run.
        </span>
      </div>
      <div className="w-filter-row">
        {(
          [
            ["models", "Model library"],
            ["experiments", "Experiments"],
            ["customization", "Customization"],
            ["compare", "Compare capabilities"],
          ] as const
        ).map(([tab, label]) => (
          <button
            key={tab}
            className={labTab === tab ? "active" : ""}
            onClick={() => setLabTab(tab)}
          >
            {label}
          </button>
        ))}
      </div>
      {labTab === "models" ? (
        <div className="w-model-grid">
          {MODELS.map((m) => (
            <article className="w-model-card" key={m.id}>
              <div className="w-model-title">
                <h3>{m.name}</h3>
                <span>
                  {m.status === "adapter-ready"
                    ? "ADAPTER AVAILABLE"
                    : m.status === "preview"
                      ? "INTERFACE PREVIEW"
                      : "RESEARCH CANDIDATE"}
                </span>
              </div>
              <p>{m.description}</p>
              <div className="w-capabilities">
                {[
                  ["Image", m.capabilities.input.image],
                  [
                    m.capabilities.runtime.controlMode === "prompt-adapted"
                      ? "Prompt-adapted controls"
                      : "Native controls",
                    m.capabilities.control.wasd,
                  ],
                  [
                    "Prompt changes",
                    m.capabilities.control.promptDuringRollout ||
                      m.capabilities.control.promptSwitching,
                  ],
                  ["Audio", m.capabilities.output.audio],
                  ["Exact resume", m.capabilities.persistence.snapshotRestore],
                ].map(([label, supported]) => (
                  <span key={String(label)} className={!supported ? "unsupported" : ""}>
                    {label}
                  </span>
                ))}
              </div>
              <p>{m.caveat}</p>
              <p>License: {m.license}</p>
              <div className="w-model-footer">
                {m.repository ? (
                  <a href={m.repository} target="_blank" rel="noreferrer">
                    Official source
                    <ArrowRight size={12} />
                  </a>
                ) : (
                  <span className="w-muted">Source verification pending</span>
                )}
                <button className="w-btn w-btn-quiet" onClick={() => onCreate(m.id)}>
                  Explore setup
                  <ArrowRight size={13} />
                </button>
              </div>
              <details style={{ marginTop: 17 }}>
                <summary className="w-muted" style={{ cursor: "pointer", fontSize: 10 }}>
                  Evidence & adapter details
                </summary>
                <ul className="w-muted" style={{ fontSize: 10, paddingLeft: 17 }}>
                  {m.evidence.map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
                <pre className="w-code-block">{JSON.stringify(m.capabilities, null, 2)}</pre>
              </details>
            </article>
          ))}
        </div>
      ) : labTab === "compare" ? (
        <>
          <section className="w-panel">
            <h2>Same idea. Different model.</h2>
            <p className="w-panel-subtitle">
              Choose model adapters and a local project to create a reproducible comparison plan. A
              plan is not a benchmark result.
            </p>
            <div className="w-character-chips">
              {MODELS.map((m) => (
                <button
                  key={m.id}
                  className={selected.includes(m.id) ? "active" : ""}
                  onClick={() =>
                    setSelected(
                      selected.includes(m.id)
                        ? selected.filter((id) => id !== m.id)
                        : [...selected, m.id],
                    )
                  }
                >
                  {selected.includes(m.id) ? <Check size={13} /> : <Plus size={13} />} {m.name}
                </button>
              ))}
            </div>
            <div className="w-table-wrap">
              <table className="w-table">
                <thead>
                  <tr>
                    <th>Adapter capability</th>
                    {comparison.map((m) => (
                      <th key={m.id}>{m.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {criteria.map(([label, fn]) => (
                    <tr key={label}>
                      <td>{label}</td>
                      {comparison.map((m) => (
                        <td key={m.id}>{fn(m) ? "Supported" : "Not exposed"}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="w-lab-toolbar">
              <label className="w-form-field" style={{ marginBottom: 0, flex: 1 }}>
                <span>SHARED INPUT</span>
                <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>
                  <option value="">Select a saved world</option>
                  {projects.map((p) => (
                    <option value={p.id} key={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="w-btn w-btn-quiet"
                disabled={!projectId || selected.length < 2}
                onClick={() => {
                  const p = projects.find((v) => v.id === projectId);
                  if (!p) return;
                  downloadBlob(
                    new Blob(
                      [
                        JSON.stringify(
                          {
                            version: 1,
                            status: "plan-not-executed",
                            projectId: p.id,
                            prompt: p.prompt,
                            seed: p.settings.seed ?? null,
                            assetIds: p.assetIds,
                            modelIds: selected,
                            providerId: p.providerId,
                            performance: p.settings.performance,
                            controlTrajectory: [],
                            metrics: [
                              "generatedFPS",
                              "latencyMs",
                              "gpuMemoryGB",
                              "actualCostUSD",
                              "visualFidelity",
                              "actionAdherence",
                              "temporalStability",
                              "spatialConsistency",
                            ],
                            notes: [
                              "Each model must receive the same input and trajectory.",
                              "Seed reproducibility is model-specific.",
                              "No benchmark has been executed by exporting this plan.",
                            ],
                          },
                          null,
                          2,
                        ),
                      ],
                      { type: "application/json" },
                    ),
                    "worlds-comparison-plan.json",
                  );
                }}
              >
                <Download size={15} />
                Export comparison plan
              </button>
            </div>
            <p className="w-muted">
              Run compatible workers with the same inputs and recorded controls. The worker
              benchmark tooling records measured performance; you can inspect recorded runs below.
            </p>
          </section>
          <section className="w-panel">
            <h2>Measured sessions</h2>
            {benchmarks.length === 0 ? (
              <p className="w-muted">
                No measured sessions yet. Connect a worker and record a session to begin collecting
                real results.
              </p>
            ) : (
              <div className="w-table-wrap">
                <table className="w-table">
                  <thead>
                    <tr>
                      <th>Run</th>
                      <th>Model</th>
                      <th>Frames</th>
                      <th>Measured FPS</th>
                      <th>Latency</th>
                      <th>Cost estimate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarks.map((b) => (
                      <tr key={b.id}>
                        <td>{b.name}</td>
                        <td>{getModel(b.modelId).name}</td>
                        <td>{b.frameCount}</td>
                        <td>{b.measuredFPS?.toFixed(1) ?? "Not measured"}</td>
                        <td>{b.latencyMs === undefined ? "Not measured" : `${b.latencyMs} ms`}</td>
                        <td>
                          {b.estimatedCostUSD === undefined
                            ? "Unavailable"
                            : `$${b.estimatedCostUSD.toFixed(3)}`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <button
              className="w-btn w-btn-quiet"
              style={{ marginTop: 20 }}
              onClick={() => setShowRaw(!showRaw)}
            >
              <SlidersHorizontal size={14} />
              {showRaw ? "Hide" : "Show"} raw registry
            </button>
            {showRaw && (
              <pre className="w-code-block" style={{ marginTop: 15 }}>
                {JSON.stringify(comparison, null, 2)}
              </pre>
            )}
          </section>
        </>
      ) : labTab === "experiments" ? (
        <ExperimentLab
          characters={characters}
          projects={projects}
          results={benchmarks}
          serverUrl={serverUrl}
          onSaved={onSaved}
        />
      ) : (
        <CustomizationPanel serverUrl={serverUrl} />
      )}
    </>
  );
}
function SettingsView({
  settings,
  onSave,
  onRefresh,
  onError,
  onToast,
}: {
  settings: AppSettings;
  onSave: (settings: AppSettings) => void;
  onRefresh: () => Promise<void>;
  onError: (message: string) => void;
  onToast: (message: string) => void;
}) {
  const [draft, setDraft] = useState(settings);
  const [token, setToken] = useState(getApiToken());
  const readiness = useComputeReadiness(settings.apiBaseUrl);
  const [busy, setBusy] = useState("");
  const [storage, setStorage] = useState<StorageEstimate>();
  useEffect(() => {
    void worldStore.estimateStorage().then(setStorage);
  }, []);
  const save = () => {
    try {
      const base = validateApiBaseUrl(draft.apiBaseUrl);
      if (
        !Number.isFinite(draft.idleTimeoutMinutes) ||
        draft.idleTimeoutMinutes < 1 ||
        draft.idleTimeoutMinutes > 240
      )
        throw new Error("Idle timeout must be between 1 and 240 minutes.");
      setApiToken(token.trim());
      onSave({ ...draft, apiBaseUrl: base });
      readiness.refresh();
      return true;
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
      return false;
    }
  };
  const test = () => {
    save();
  };
  return (
    <>
      <PageHeading
        overline="YOUR WORLDS, ON YOUR TERMS"
        title="Make yourself at home."
        description="Connect your compute. Keep control of your library."
      />
      <div className="w-settings-layout">
        <ComputeInventory
          key={settings.apiBaseUrl}
          serverUrl={settings.apiBaseUrl}
          initialProvider={settings.defaultProvider}
        />
        <BillingPanel key={`billing:${settings.apiBaseUrl}`} serverUrl={settings.apiBaseUrl} />
        <section className="w-panel">
          <div className="w-settings-title">
            <Radio size={19} />
            <h2>Session manager</h2>
          </div>
          <p className="w-panel-subtitle">
            Your browser connects to a session manager, which handles GPU providers and worker
            credentials. Provider API keys belong in the manager’s environment, never in the
            browser.
          </p>
          <label className="w-form-field">
            <span>SERVER URL</span>
            <input
              type="url"
              value={draft.apiBaseUrl}
              placeholder="Same origin (default) or http://localhost:8788"
              onChange={(e) => setDraft({ ...draft, apiBaseUrl: e.target.value })}
            />
            <small>
              Leave blank for this site’s /api/v1/worlds endpoint. Remote connections require HTTPS.
              Use an approved local or self-hosted session manager.
            </small>
          </label>
          <label className="w-form-field">
            <span>SESSION MANAGER ACCESS TOKEN</span>
            <input
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Optional manager token · not your provider API key"
            />
            <small>Kept only for this browser tab’s session. Excluded from library exports.</small>
          </label>
          <div className="w-settings-bottom">
            <button className="w-btn w-btn-primary" onClick={save}>
              <Check size={15} />
              Save connection
            </button>
            <button className="w-btn w-btn-quiet" disabled={readiness.checking} onClick={test}>
              {readiness.checking ? <Loader2 size={15} className="w-spin" /> : <Radio size={15} />}
              Test connection
            </button>
            <span className="w-muted" style={{ fontSize: 10 }}>
              {readiness.checking
                ? "Checking readiness…"
                : readiness.error
                  ? "Connection unavailable"
                  : "Session manager connected"}
            </span>
          </div>
          {readiness.error && (
            <p className="w-inline-error" role="alert">
              {readiness.error}
            </p>
          )}
        </section>
        <section className="w-panel">
          <div className="w-settings-title">
            <Monitor size={19} />
            <h2>Compute, your way</h2>
          </div>
          <p className="w-panel-subtitle">
            RunPod is the default GPU service for Worlds. Modal remains an alternative; Earth’s GPU
            infrastructure is separate.
          </p>
          <div className="w-two-col">
            <label className="w-form-field">
              <span>DEFAULT PROVIDER</span>
              <select
                value={draft.defaultProvider}
                onChange={(e) =>
                  setDraft({ ...draft, defaultProvider: e.target.value as ProviderId })
                }
              >
                {PROVIDERS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="w-form-field">
              <span>IDLE TIMEOUT · MINUTES</span>
              <input
                type="number"
                min={1}
                max={240}
                value={draft.idleTimeoutMinutes}
                onChange={(e) => setDraft({ ...draft, idleTimeoutMinutes: Number(e.target.value) })}
              />
              <small>
                Ends idle sessions while the player is open. Disconnected workers also apply their
                own timeout.
              </small>
            </label>
          </div>
          <label className="w-check-label">
            <input
              type="checkbox"
              checked={draft.retainWorker}
              onChange={(e) => setDraft({ ...draft, retainWorker: e.target.checked })}
            />
            Retain my worker after a session, when supported
          </label>
          <p className="w-muted" style={{ fontSize: 10 }}>
            Retained GPU workers may continue to incur charges. The manager controls actual worker
            lifecycle and retention.
          </p>
          <div className="w-provider-list">
            {PROVIDERS.map((p) => {
              const live = readiness.report?.providers.find((x) => x.id === p.id);
              return (
                <div key={p.id} className={`w-provider-row ${live?.configured ? "ready" : ""}`}>
                  <span className="w-status-dot" />
                  {p.name}
                  <span>
                    {live?.configured ? "CONFIGURED" : live ? "NOT CONFIGURED" : "NOT CHECKED"}
                  </span>
                </div>
              );
            })}
          </div>
          <button className="w-btn w-btn-quiet" style={{ marginTop: 18 }} onClick={save}>
            Save preferences
            <Check size={14} />
          </button>
        </section>
        <section className="w-panel">
          <div className="w-settings-title">
            <CheckCircle2 size={19} />
            <h2>Launch readiness</h2>
          </div>
          <p className="w-panel-subtitle">
            Read-only checks verify your selected provider and installed models. Checking readiness
            never allocates a GPU.
          </p>
          <ReadinessRefresh checking={readiness.checking} onRefresh={readiness.refresh} />
          {readiness.report && (
            <ReadinessDetails report={readiness.report} providerId={draft.defaultProvider} />
          )}
          {readiness.error && <p className="w-inline-error">{readiness.error}</p>}
        </section>
        <WorkerManager serverUrl={settings.apiBaseUrl} />
        <section className="w-panel">
          <div className="w-settings-title">
            <ShieldCheck size={19} />
            <h2>Your local library</h2>
          </div>
          <p className="w-panel-subtitle">
            Projects, media, characters, scenes, and recordings are saved in this browser’s
            IndexedDB. Clearing site data removes them. Export a backup to keep your work elsewhere.
          </p>
          {storage && (
            <p className="w-muted" style={{ marginBottom: 18, fontSize: 11 }}>
              {((storage.usage ?? 0) / 1024 / 1024).toFixed(1)} MB used by this origin
              {storage.quota
                ? ` · approximately ${Math.round(storage.quota / 1024 / 1024)} MB available quota`
                : ""}
            </p>
          )}
          <div className="w-settings-bottom">
            <button
              className="w-btn w-btn-quiet"
              disabled={!!busy}
              onClick={() => {
                void (async () => {
                  setBusy("export");
                  try {
                    downloadBlob(
                      await worldStore.exportLibrary(),
                      `worlds-library-${new Date().toISOString().slice(0, 10)}.json`,
                    );
                    onToast("Library backup exported.");
                  } catch (e) {
                    onError(String(e));
                  } finally {
                    setBusy("");
                  }
                })();
              }}
            >
              {busy === "export" ? (
                <Loader2 size={15} className="w-spin" />
              ) : (
                <Download size={15} />
              )}
              Export library
            </button>
            <label className="w-btn w-btn-quiet" style={{ cursor: "pointer" }}>
              <Upload size={15} />
              Import library
              <input
                className="w-hidden-input"
                type="file"
                accept="application/json,.json"
                disabled={!!busy}
                onChange={(e) => {
                  void (async () => {
                    const file = e.target.files?.[0];
                    e.target.value = "";
                    if (!file) return;
                    setBusy("import");
                    try {
                      const count = await worldStore.importLibrary(file);
                      await onRefresh();
                      onToast(`Imported ${count} library records.`);
                    } catch (e) {
                      onError(e instanceof Error ? e.message : String(e));
                    } finally {
                      setBusy("");
                    }
                  })();
                }}
              />
            </label>
            <button
              className="w-btn w-btn-quiet"
              onClick={() => {
                void (async () => {
                  try {
                    const ok = await worldStore.requestPersistentStorage();
                    onToast(
                      ok
                        ? "Persistent storage enabled for this browser."
                        : "Browser did not grant persistent storage. Keep a backup of important work.",
                    );
                  } catch (e) {
                    onError(String(e));
                  }
                })();
              }}
            >
              <ShieldCheck size={15} />
              Protect local storage
            </button>
          </div>
          <p className="w-muted" style={{ marginTop: 16, fontSize: 10 }}>
            Imports merge into your library. Records with matching IDs are replaced after the
            archive is validated. Credentials are never exported.
          </p>
        </section>
        <section className="w-panel">
          <h2>Connect a GPU worker</h2>
          <p className="w-panel-subtitle">
            Run the session manager locally, configure its RunPod endpoint or a compatible worker
            gateway, and select the same model in the composer. The model gateway protocol keeps
            inference separate from cloud provisioning.
          </p>
          <pre className="w-code-block">
            {
              "Browser → Worlds session manager → RunPod / local / other GPU worker\n\nRequired server-side setup:\n• Model gateway URL for the selected provider\n• Provider credentials for provisioning, if needed\n• Session manager access token\n\nOptional prompt intelligence:\nWORLDS_LLM_BASE_URL\nWORLDS_LLM_MODEL\nWORLDS_LLM_API_KEY"
            }
          </pre>
          <p className="w-muted" style={{ marginTop: 15, fontSize: 10 }}>
            Worlds does not provision a GPU until you explicitly start a live session. Interaction
            previews work without credentials or GPU access.
          </p>
        </section>
      </div>
    </>
  );
}
