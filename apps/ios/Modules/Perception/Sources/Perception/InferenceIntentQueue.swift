/// User taps are intents, not camera frames: preserve their order across a busy Vision call.
/// The caller owns synchronization; overflow is explicit rather than silently replacing a tap.
public struct InferenceIntentQueue<Element: Sendable>: Sendable {
  private var values: [Element] = []
  public let capacity: Int
  public init(capacity: Int = PerceptionLimits.maximumPendingTaps) {
    self.capacity = max(0, capacity)
  }
  public var isEmpty: Bool { values.isEmpty }
  public mutating func offer(_ element: Element) -> Bool {
    guard values.count < capacity else { return false }
    values.append(element)
    return true
  }
  public mutating func take() -> Element? { values.isEmpty ? nil : values.removeFirst() }
  public mutating func clear() { values.removeAll() }
}
