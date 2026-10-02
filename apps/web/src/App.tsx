import { MotionConfig } from "motion/react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { AppProviders } from "@/app/providers";
import { useApplyTheme } from "@/app/theme";
import { CesiumViewport } from "@/cesium/CesiumViewport";
import { SceneBridge } from "@/cesium/SceneBridge";
import { AppShell } from "@/features/shell/AppShell";
import { useSettings } from "@/state/settings";

function ThemedApp() {
  useApplyTheme();
  // Every animation follows the operating system's setting, and the app's own switch
  // turns them off regardless of it.
  const reducedMotion = useSettings((s) => s.reducedMotion);
  return (
    <MotionConfig reducedMotion={reducedMotion ? "always" : "user"}>
      <div className="app">
        <CesiumViewport />
        <SceneBridge />
        <ErrorBoundary>
          <AppShell />
        </ErrorBoundary>
      </div>
    </MotionConfig>
  );
}

export function App() {
  return (
    <ErrorBoundary>
      <AppProviders>
        <ThemedApp />
      </AppProviders>
    </ErrorBoundary>
  );
}
