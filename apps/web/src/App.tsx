import { ErrorBoundary } from "@/app/ErrorBoundary";
import { AppProviders } from "@/app/providers";
import { useApplyTheme } from "@/app/theme";
import { CesiumViewport } from "@/cesium/CesiumViewport";
import { SceneBridge } from "@/cesium/SceneBridge";
import { AppShell } from "@/features/shell/AppShell";

function ThemedApp() {
  useApplyTheme();
  return (
    <div className="app">
      <CesiumViewport />
      <SceneBridge />
      <ErrorBoundary>
        <AppShell />
      </ErrorBoundary>
    </div>
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
