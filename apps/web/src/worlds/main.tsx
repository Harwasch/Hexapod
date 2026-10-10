import { Component, type ReactNode } from "react";
import { createRoot } from "react-dom/client";

import WorldsApp from "./WorldsApp";

const container = document.getElementById("root");
if (!container) throw new Error("#root element missing");

class WorldsBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  override state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  override render() {
    if (this.state.failed) {
      return (
        <main style={{ maxWidth: 540, margin: "15vh auto", padding: 32 }}>
          <p>WORLDS</p>
          <h1>Something interrupted the explorer.</h1>
          <p>Your saved library is still on this device. Reload to try again.</p>
          <button onClick={() => window.location.reload()}>Reload Worlds</button>
          <p>If a cloud worker was running, check its status in Settings after reloading.</p>
        </main>
      );
    }
    return this.props.children;
  }
}

createRoot(container).render(
  // A player mount starts a paid worker. Development StrictMode's effect replay must
  // not duplicate that externally observable operation.
  <WorldsBoundary>
    <WorldsApp />
  </WorldsBoundary>,
);
