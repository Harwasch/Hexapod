import Capture
import Foundation

/// The synchronous ingress reserves capacity BEFORE creating a task. Capacity counts
/// queued + in-flight jobs. One actor drains FIFO; overflow is explicit (never drops
/// just depth while keeping the corresponding video frame). close() drains accepted jobs.
final class BackgroundWriter<Job: Sendable>: Sendable {
  private struct State {
    var queue: [Job] = []
    var outstanding = 0
    var running = false
    var closed = false
    var error: (any Error)?
    var waiters: [CheckedContinuation<Void, Never>] = []
  }
  private let lock = NSLock()
  private nonisolated(unsafe) var state = State()
  private let capacity: Int
  private let worker: Worker
  init(capacity: Int, write: @escaping @Sendable (Job) async throws -> Void) {
    precondition(capacity > 0)
    self.capacity = capacity
    worker = Worker(write: write)
  }
  var outstanding: Int { lock.withLock { state.outstanding } }
  @discardableResult func submit(_ job: Job) -> Bool {
    let result: (Bool, Bool) = lock.withLock {
      guard !state.closed, state.error == nil, state.outstanding < capacity else {
        return (false, false)
      }
      state.queue.append(job)
      state.outstanding += 1
      let start = !state.running
      state.running = true
      return (true, start)
    }
    if result.1 { Task { await worker.drain(self) } }
    return result.0
  }
  func close() async throws {
    await withCheckedContinuation { continuation in
      let resume = lock.withLock {
        state.closed = true
        if state.outstanding == 0 { return true }
        state.waiters.append(continuation)
        return false
      }
      if resume { continuation.resume() }
    }
    if let error = lock.withLock({ state.error }) { throw error }
  }
  private func next() -> Job? {
    lock.withLock {
      guard !state.queue.isEmpty else {
        state.running = false
        return nil
      }
      return state.queue.removeFirst()
    }
  }
  private func completed(error: (any Error)? = nil) {
    let waiters: [CheckedContinuation<Void, Never>] = lock.withLock {
      state.outstanding -= 1
      if let error {
        state.error = error
        state.outstanding -= state.queue.count
        state.queue.removeAll()
        state.closed = true
      }
      guard state.outstanding == 0 else { return [] }
      let waiters = state.waiters
      state.waiters = []
      return waiters
    }
    waiters.forEach { $0.resume() }
  }
  private actor Worker {
    let write: @Sendable (Job) async throws -> Void
    init(write: @escaping @Sendable (Job) async throws -> Void) { self.write = write }
    func drain(_ owner: BackgroundWriter<Job>) async {
      while let job = owner.next() {
        do {
          try await write(job)
          owner.completed()
        } catch {
          owner.completed(error: error)
          return
        }
      }
    }
  }
}

struct RecordedFile: Sendable {
  let name: String
  let data: Data
}
enum RecordingPacket: Sendable {
  case frame(CaptureFrame, [RecordedFile])
  case location(CaptureLocation)
}

/// CaptureBundleWriter is created and used only here. All directory, depth, JSONL,
/// synchronize and manifest writes occur on this background actor, never MainActor.
actor RecordingDiskWriter {
  private let bundle: CaptureBundleWriter
  let directory: URL
  init(directory: URL, device: String) throws {
    self.directory = directory
    bundle = try CaptureBundleWriter(directory: directory, device: device)
  }
  static func create(directory: URL, device: String) async throws -> RecordingDiskWriter {
    // An actor initializer executes on its caller; detach creation as well as writes.
    try await Task.detached { try RecordingDiskWriter(directory: directory, device: device) }.value
  }
  func write(_ packet: RecordingPacket) throws {
    switch packet {
    case .frame(let frame, let files):
      for file in files {
        guard CaptureBundleWriter.safeName(file.name) else { throw CaptureError.invalidFile }
        try file.data.write(to: directory.appendingPathComponent(file.name))
      }
      try bundle.append(frame)
    case .location(let fix): try bundle.appendLocation(fix)
    }
  }
  func finish() throws { _ = try bundle.finish() }
}
