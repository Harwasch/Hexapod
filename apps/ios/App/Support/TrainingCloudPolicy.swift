enum TrainingCloudPolicy {
  enum Decision { case skip, fetch, stop }
  static func decision(busy: Bool, status: String?, failed: Bool) -> Decision {
    if status == "complete" || status == "failed" || status == "error" || failed { return .stop }
    return busy ? .skip : .fetch
  }
}
