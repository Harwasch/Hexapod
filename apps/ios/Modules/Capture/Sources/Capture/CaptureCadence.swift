import Foundation

/// Constant-space admission on the serial frame delegate. No catch-up burst after a stall.
/// Quantize elapsed time to nanoseconds: the clock's resolution, not a fitted tolerance.
public struct CaptureCadence: Sendable {
  public let framesPerSecond: Int
  private var lastSlot: Int64 = -1
  private var origin: Double?
  public init(framesPerSecond: Int = 15) {
    precondition((1...60).contains(framesPerSecond))
    self.framesPerSecond = framesPerSecond
  }
  public mutating func admit(_ timestamp: Double) -> Bool {
    guard timestamp.isFinite, timestamp >= 0 else { return false }
    if origin == nil { origin = timestamp }
    let elapsed = timestamp - origin!
    guard elapsed >= 0, elapsed < 1e8 else { return false }
    let nanoseconds = Int64((elapsed * 1_000_000_000).rounded())
    // Round the rational slot boundary to the same nanosecond resolution.
    let slot = (nanoseconds * Int64(framesPerSecond) + Int64(framesPerSecond / 2)) / 1_000_000_000
    guard slot > lastSlot else { return false }
    lastSlot = slot
    return true
  }
}
