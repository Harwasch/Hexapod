import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/api/client";
import { LandInspectionForm } from "@/features/land/LandInspectionForm";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
it("records observation time, measurement units, and a stable retry key", async () => {
  const post = vi
    .spyOn(api, "POST")
    .mockRejectedValueOnce(new Error("Network unavailable"))
    .mockResolvedValue({ data: {}, response: new Response() });
  const saved = vi.fn().mockResolvedValue(undefined);
  render(<LandInspectionForm landId="land" featureId="pole" onSaved={saved} />);
  fireEvent.change(screen.getByLabelText("Inspection notes"), {
    target: { value: "Corrosion at base" },
  });
  fireEvent.change(screen.getByLabelText("Observed at (UTC)"), {
    target: { value: "2026-10-09T12:30" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Add measurement" }));
  for (const [field, value] of Object.entries({ name: "height", value: "21.5", unit: "m" })) {
    fireEvent.change(screen.getByLabelText(`Measurement 1 ${field}`), { target: { value } });
  }
  fireEvent.click(screen.getByRole("button", { name: "Record inspection" }));
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Record inspection" }));
  await waitFor(() => expect(saved).toHaveBeenCalledOnce());
  const calls = post.mock.calls as unknown as [string, { body: unknown }][];
  expect(calls[0]?.[1].body).toMatchObject({
    observedAt: "2026-10-09T12:30:00.000Z",
    measurements: { height: 21.5 },
    measurementUnits: { height: "m" },
  });
  expect(calls[0]?.[1].body).toEqual(calls[1]?.[1].body);
  expect(screen.getByLabelText("Inspection notes")).toHaveValue("");
});

it("rejects duplicate measurement names without submitting", async () => {
  const post = vi.spyOn(api, "POST");
  render(<LandInspectionForm landId="land" featureId="pole" onSaved={() => Promise.resolve()} />);
  fireEvent.change(screen.getByLabelText("Inspection notes"), { target: { value: "Inspection" } });
  for (const index of [1, 2]) {
    fireEvent.click(screen.getByRole("button", { name: "Add measurement" }));
    for (const [field, value] of Object.entries({ name: "height", value: "20", unit: "m" })) {
      fireEvent.change(screen.getByLabelText(`Measurement ${index} ${field}`), {
        target: { value },
      });
    }
  }
  fireEvent.click(screen.getByRole("button", { name: "Record inspection" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("different name");
  expect(post).not.toHaveBeenCalled();
});
