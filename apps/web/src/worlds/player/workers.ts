export interface ReusableWorker {
  id: string;
  provider: string;
  modelId?: string;
  status: string;
  managed?: boolean;
  estimatedHourlyCost?: number | null;
}
export interface WorkerSession {
  id: string;
  workerId: string;
  status: string;
}

/** Failed or uncertain sessions are still considered occupied until explicitly cleaned up. */
export function selectRetainedWorker(
  workers: ReusableWorker[],
  sessions: WorkerSession[],
  provider: string,
  modelId?: string,
): ReusableWorker | undefined {
  const occupied = new Set(
    sessions.filter((session) => session.status !== "stopped").map((session) => session.workerId),
  );
  const candidates = workers.filter(
    (worker) =>
      worker.provider === provider &&
      (!modelId ||
        worker.modelId === modelId ||
        (!worker.modelId && modelId === "astronex-world")) &&
      worker.managed === true &&
      ["ready", "starting"].includes(worker.status) &&
      !occupied.has(worker.id),
  );
  return candidates.find((worker) => worker.status === "ready") ?? candidates[0];
}
