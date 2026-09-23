import { Component, type ErrorInfo, type ReactNode } from "react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { createLogger } from "@/lib/log";

const log = createLogger("ui");

interface Props {
  children: ReactNode;
  /** Compact fallback for a single panel instead of the whole app. */
  inline?: boolean;
  label?: string;
}

interface State {
  error: Error | null;
}

/** Catches render errors so a broken panel never takes the globe down with it. */
export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    log.error("render error", {
      label: this.props.label,
      error: error.message,
      stack: info.componentStack ?? undefined,
    });
  }

  private reset = () => this.setState({ error: null });

  override render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;
    if (this.props.inline) {
      return (
        <GlassPanel padding="sm" compact role="alert">
          <p className="glass-title">{this.props.label ?? "This panel"} hit a problem</p>
          <p className="glass-muted" style={{ fontSize: "var(--text-xs)" }}>
            {error.message}
          </p>
          <GlassButton size="sm" onClick={this.reset}>
            Try again
          </GlassButton>
        </GlassPanel>
      );
    }
    return (
      <div className="fatal" role="alert">
        <GlassPanel strong className="fatal__card">
          <h1>Something went wrong</h1>
          <p>The interface failed to render. The world may still be running underneath.</p>
          <pre>{error.message}</pre>
          <GlassButton variant="primary" onClick={() => window.location.reload()}>
            Reload
          </GlassButton>
        </GlassPanel>
      </div>
    );
  }
}
