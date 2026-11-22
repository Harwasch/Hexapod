import type { BridgeMessage } from "./generated";

/** Native gets a useful state without leaking signed tile URLs or script stacks. */
export function installErrorForwarding(
  target: Window,
  emit: (message: BridgeMessage) => void,
): () => void {
  const error = (): void => emit({ version: 1, type: "failure", message: "Globe script failed" });
  const rejection = (): void =>
    emit({ version: 1, type: "failure", message: "Globe request failed" });
  target.addEventListener("error", error);
  target.addEventListener("unhandledrejection", rejection);
  return () => {
    target.removeEventListener("error", error);
    target.removeEventListener("unhandledrejection", rejection);
  };
}
