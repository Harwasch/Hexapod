import type { ModelCapabilities, ModelInfo, ProviderInfo } from "./types";
import runtimeCapabilities from "./runtimeCapabilities.json";
const runtimes = runtimeCapabilities as Record<
  string,
  ModelCapabilities & { nativeActions: string[] }
>;
export function emptyCapabilities(): ModelCapabilities {
  return {
    input: { text: false, image: false, multiImage: false, video: false, audio: false },
    control: {
      wasd: false,
      mouseLook: false,
      camera6DoF: false,
      gamepad: false,
      discreteActions: false,
      continuousActions: false,
      semanticActions: false,
      promptDuringRollout: false,
      promptSwitching: false,
      timedEvents: false,
      characterReference: false,
    },
    output: { video: false, audio: false, depth: false, cameraPose: false },
    persistence: { nativeMemory: false, snapshotRestore: false, deterministicSeed: false },
    runtime: { resolutionOptions: [], realtime: false },
    customization: { lora: false, fineTune: false, adapters: false },
  };
}
const astronex = runtimes["astronex-world"] ?? emptyCapabilities();
/** Catalog capabilities describe this application's adapter, not unverified paper claims.
 * Worker-reported capabilities take precedence while a session is active. */
export const MODELS: ModelInfo[] = [
  {
    id: "astronex-world",
    name: "Astronex World",
    family: "Action-conditioned video",
    description:
      "Open-weight world generation with text and image conditioning. Our adapter supports combined movement and camera controls between generated blocks.",
    repository: "https://github.com/Astronex-Robotics/Astronex-World",
    license: "Apache-2.0",
    status: "adapter-ready",
    capabilities: astronex,
    nativeActions: runtimes["astronex-world"]?.nativeActions ?? [],
    evidence: [
      "Official repository: Astronex-Robotics/Astronex-World",
      "Checkpoint: Astronex-Lab/Astronex-World",
    ],
    caveat:
      "GPU validation pending. Combined keys, mouse, and analog input are sampled between generation blocks. Persistent KV memory and realtime response are not established.",
  },
  {
    id: "ltx-2.5",
    name: "LTX 2.5",
    family: "Continuous audio-visual exploration",
    description:
      "Explore an ongoing generated scene with text, voice, and prompt-adapted walking controls. Video and audio context continue across overlapping chunks.",
    repository: "https://github.com/Lightricks/LTX-2",
    license: "LTX-2.5 community license; Gemma text encoder terms",
    status: "adapter-ready",
    capabilities: runtimes["ltx-2.5"] ?? emptyCapabilities(),
    nativeActions: runtimes["ltx-2.5"]?.nativeActions ?? [],
    evidence: ["Official LTX-2.5 distilled pipeline and latent-carry chunk operators"],
    caveat:
      "GPU validation pending. Walking and mouse input become camera prompts for the next chunk. Continuity, exact navigation, realtime latency, and cost are unverified. Session time and budget limits still apply.",
  },
  {
    id: "matrix-game-3",
    name: "Matrix-Game 3",
    family: "Interactive world model",
    description:
      "Image-and-text-conditioned Unreal environments using the pinned Matrix-Game 3 inference pipeline.",
    repository: "https://github.com/SkyworkAI/Matrix-Game",
    license: "Apache-2.0 (v3; verify checkpoint terms)",
    status: "adapter-ready",
    capabilities: runtimes["matrix-game-3"] ?? emptyCapabilities(),
    nativeActions: runtimes["matrix-game-3"]?.nativeActions ?? [],
    evidence: ["Official repository: SkyworkAI/Matrix-Game"],
    caveat:
      "Requires a starting image and a dedicated GPU runtime. Discrete controls and prompts apply between clips; GPU quality, speed, and continuation consistency are unvalidated.",
  },
  {
    id: "lingbot-world",
    name: "LingBot-World",
    family: "Interactive world model",
    description:
      "Research candidate for interactive visual environments and long-horizon world generation.",
    repository: "https://github.com/Robbyant/lingbot-world",
    license: "Review upstream checkpoint terms",
    status: "research",
    capabilities: emptyCapabilities(),
    nativeActions: [],
    evidence: [],
    caveat:
      "Integration and hardware requirements need verification. No inference or performance claims are made here.",
  },
  {
    id: "alaya-world",
    name: "AlayaWorld",
    family: "Interactive world model",
    description: "Research candidate with released inference code and model weights.",
    license: "Research / noncommercial; gated dependency",
    status: "research",
    capabilities: emptyCapabilities(),
    nativeActions: [],
    evidence: ["LTX-2 derived checkpoint; gated Gemma dependency"],
    caveat:
      "Checkpoint terms and gated dependencies must be reviewed. No validated adapter is enabled.",
  },
  {
    id: "sana-wm",
    name: "SANA-WM",
    family: "Streaming world model",
    description:
      "Image-and-text-conditioned camera exploration using the official SANA-WM streaming pipeline.",
    repository: "https://github.com/NVlabs/Sana",
    license: "Review code and checkpoint terms",
    status: "adapter-ready",
    capabilities: runtimes["sana-wm"] ?? emptyCapabilities(),
    nativeActions: runtimes["sana-wm"]?.nativeActions ?? [],
    evidence: [
      "Official asset/docs/sana_wm.md",
      "Checkpoint: Efficient-Large-Model/SANA-WM_streaming",
    ],
    caveat:
      "Requires a starting image and the SANA, LTX, and Gemma checkpoint components. Adapter continues between clips; GPU validation and license review for your intended use remain necessary.",
  },
  {
    id: "forge-wm",
    name: "ForgeWM",
    family: "Action-conditioned video",
    description: "Image-conditioned game environments with discrete keyboard and camera actions.",
    repository: "https://github.com/asdfo123/ForgeWM",
    license: "Apache-2.0",
    status: "adapter-ready",
    capabilities: runtimes["forge-wm"] ?? emptyCapabilities(),
    nativeActions: runtimes["forge-wm"]?.nativeActions ?? [],
    evidence: ["Official repository: asdfo123/ForgeWM", "Checkpoint: ForgeWM/ForgeWM"],
    caveat:
      "Requires a starting image. Text prompts do not condition this model; movement applies between clips. GPU execution and performance are unvalidated.",
  },
  {
    id: "helix-world",
    name: "HelixWorld",
    family: "Audio-visual world model",
    description:
      "Generate an audio-visual clip from an image and scene description, with camera direction.",
    repository: "https://github.com/NoizAI/HelixWorld",
    license: "Apache-2.0; Gemma text encoder terms",
    status: "adapter-ready",
    capabilities: runtimes["helix-world"] ?? emptyCapabilities(),
    nativeActions: runtimes["helix-world"]?.nativeActions ?? [],
    evidence: ["Checkpoint: NoizAI/HelixWorld-preview", "Official preview exposes an offline CLI"],
    caveat:
      "Offline 121-frame clips with generated audio; resume explicitly starts the next visual continuation. Requires an 80 GB-class GPU and licensed Gemma artifacts. GPU inference is unvalidated.",
  },
  {
    id: "hy-worldplay",
    name: "HY-WorldPlay",
    family: "Interactive world model",
    description: "Candidate for evaluation in the world-model laboratory.",
    repository: "https://github.com/Tencent-Hunyuan/HY-WorldPlay",
    license: "Review upstream model license",
    status: "research",
    capabilities: emptyCapabilities(),
    nativeActions: [],
    evidence: [],
    caveat:
      "Release, checkpoint terms, and inference interface require verification before integration.",
  },
  {
    id: "oasis",
    name: "Oasis",
    family: "Action-conditioned video",
    description: "Research candidate for action-conditioned generated environments.",
    repository: "https://github.com/etched-ai/open-oasis",
    license: "Review code and weight licenses separately",
    status: "research",
    capabilities: emptyCapabilities(),
    nativeActions: [],
    evidence: [],
    caveat:
      "No validated adapter is enabled. Requires model-specific controls and checkpoint review.",
  },
  {
    id: "diamond",
    name: "DIAMOND",
    family: "Diffusion world model",
    description:
      "Research candidate for learning environments from visual observations and actions.",
    repository: "https://github.com/eloialonso/diamond",
    license: "Review checkpoint-specific terms",
    status: "research",
    capabilities: emptyCapabilities(),
    nativeActions: [],
    evidence: [],
    caveat:
      "No validated adapter is enabled. Task-specific checkpoints are not general text-to-world models.",
  },
];
export function getModel(id: string): ModelInfo {
  return (
    MODELS.find((model) => model.id === id) ?? {
      id,
      name: id,
      description: "Custom model provided by a configured worker.",
      license: "Unverified",
      status: "research",
      capabilities: emptyCapabilities(),
      nativeActions: [],
      evidence: [],
      caveat: "Capabilities must be reported by the configured worker.",
    }
  );
}
export const PROVIDERS: ProviderInfo[] = [
  {
    id: "runpod",
    name: "RunPod",
    configured: false,
    canProvision: false,
    message:
      "Primary cloud provider. Configure credentials and a worker template on the session manager.",
  },
  {
    id: "local",
    name: "Local GPU",
    configured: false,
    canProvision: false,
    message:
      "Connect your local model gateway. Prompts and media remain on your machine when all services are local.",
  },
  {
    id: "modal",
    name: "Modal",
    configured: false,
    canProvision: false,
    message:
      "Use a separately approved Worlds Sandbox runtime or an existing gateway. Earth GPU tasks remain independent.",
  },
  {
    id: "lambda",
    name: "Lambda",
    configured: false,
    canProvision: false,
    message:
      "Use an approved GPU image and HTTPS gateway bootstrap configured on the session manager.",
  },
  ...(["coreweave", "aws", "gcp", "azure"] as const).map((id) => ({
    id,
    name: {
      lambda: "Lambda",
      coreweave: "CoreWeave",
      aws: "AWS",
      gcp: "Google Cloud",
      azure: "Azure",
    }[id],
    configured: false,
    canProvision: false,
    message: "Provider extension point; not integrated yet.",
  })),
];
export function supportsLivePrompt(capabilities: ModelCapabilities): boolean {
  return (
    capabilities.control.promptDuringRollout ||
    capabilities.control.promptSwitching ||
    capabilities.control.semanticActions ||
    capabilities.control.timedEvents
  );
}
