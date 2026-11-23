import { expect, it, vi } from "vitest";
import { installErrorForwarding } from "./errors";

it("forwards JS errors and rejected promises without URLs or stacks, then detaches", () => {
  const emit = vi.fn();
  const remove = installErrorForwarding(window, emit);
  window.dispatchEvent(new ErrorEvent("error", { message: "secret URL" }));
  const rejection = new Event("unhandledrejection");
  window.dispatchEvent(rejection);
  expect(emit.mock.calls).toEqual([
    [{ version: 1, type: "failure", message: "Globe script failed" }],
    [{ version: 1, type: "failure", message: "Globe request failed" }],
  ]);
  remove();
  window.dispatchEvent(new ErrorEvent("error"));
  expect(emit).toHaveBeenCalledTimes(2);
});
