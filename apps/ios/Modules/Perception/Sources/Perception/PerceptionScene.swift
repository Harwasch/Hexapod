import EntitySchema
import Foundation

public struct SceneSelection: Sendable {
  public var entities: [Entity]
  public var masks: [SubjectMask]
  public var viewportToImage: [Double] = [1, 0, 0, 1, 0, 0]
  public var capturedAt: TimeInterval = 0
  public init(
    entities: [Entity], masks: [SubjectMask], viewportToImage: [Double] = [1, 0, 0, 1, 0, 0],
    capturedAt: TimeInterval = 0
  ) {
    self.entities = entities
    self.masks = masks
    self.viewportToImage = viewportToImage
    self.capturedAt = capturedAt
  }
}

/// All inference and tracker state live on this actor; ARKit-owned buffers are copied first.
public actor PerceptionScene {
  private let segmenter: any Segmenter
  private let instances: any InstanceSegmenting
  private let identifier: any ImageIdentifying
  private var snapshot: FrameSnapshot?
  private var selected: Entity?
  private var tracker = EntityTracker()
  private var revision = 0
  private var queriesInFlight = 0
  private var viewportToImage: [Double] = [1, 0, 0, 1, 0, 0]
  private var capturedAt: TimeInterval = 0
  private var latest = SceneSelection(entities: [], masks: [])

  public init(
    segmenter: any Segmenter,
    instances: any InstanceSegmenting,
    identifier: any ImageIdentifying
  ) {
    self.segmenter = segmenter
    self.instances = instances
    self.identifier = identifier
  }
  public func update(_ frame: FrameSnapshot, viewportToImage: [Double], capturedAt: TimeInterval) {
    snapshot = frame
    self.viewportToImage = viewportToImage
    self.capturedAt = capturedAt
  }
  public func clear() {
    revision += 1
    snapshot = nil
    selected = nil
    tracker = EntityTracker()
    latest = SceneSelection(entities: [], masks: [])
  }
  public func selection() -> SceneSelection { latest }

  public func select(_ point: ImagePoint) async throws -> SceneSelection {
    queriesInFlight += 1
    defer { queriesInFlight -= 1 }
    revision += 1
    let id = revision
    let transform = viewportToImage
    let time = capturedAt
    guard let frame = snapshot else { throw PerceptionError.insufficientDepth }
    let mask = try await segmenter.segment(frame.image, prompt: .point(point))
    try Task.checkCancellation()
    let label = try await identifier.identify(frame.image, mask: mask)
    try Task.checkCancellation()
    guard id == revision else { throw CancellationError() }
    var entity = try MaskLifter.lift(
      mask: mask, depth: frame.depth, camera: frame.camera, label: label?.label ?? "subject")
    if let label {
      entity.attrs["identifiedOnDevice"] = .bool(true)
      entity.attrs["identificationConfidence"] = .number(label.confidence)
    }
    selected = entity
    latest = SceneSelection(
      entities: [entity], masks: [mask], viewportToImage: transform, capturedAt: time)
    tracker = EntityTracker()
    _ = tracker.update([Detection(entity: entity, bounds: mask.bounds!)], camera: frame.camera)
    return latest
  }

  public func refreshSelection(isQuery: Bool = false) async throws -> SceneSelection {
    guard isQuery || queriesInFlight == 0 else { return latest }
    let transform = viewportToImage
    let time = capturedAt
    guard let frame = snapshot else { return latest }
    let id = revision
    var entities: [Entity] = []
    var masks: [SubjectMask] = []
    let candidates = tracker.tracks.isEmpty ? latest.entities : tracker.tracks.map(\.entity)
    for previous in candidates {
      guard let point = frame.camera.project(previous.anchor.world),
        (0..<1).contains(point.x), (0..<1).contains(point.y)
      else { continue }
      let mask: SubjectMask
      do {
        mask = try await segmenter.segment(frame.image, prompt: .point(point))
      } catch is CancellationError { throw CancellationError() } catch { continue }
      try Task.checkCancellation()
      guard
        var entity = try? MaskLifter.lift(
          mask: mask, depth: frame.depth, camera: frame.camera,
          label: previous.label)
      else { continue }
      let p = entity.anchor.world
      let q = previous.anchor.world
      let distance = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2) + pow(p.z - q.z, 2))
      let overlap = frame.camera.projectedBounds(previous)?.iou(mask.bounds!) ?? 0
      guard distance <= tracker.maximumDistance,
        overlap >= 0.05 || distance <= tracker.maximumDistance * 0.5
      else { continue }
      entity.attrs.merge(previous.attrs) { current, _ in current }
      if masks.contains(where: { ($0.bounds?.iou(mask.bounds!) ?? 0) > 0.7 }) { continue }
      entities.append(entity)
      masks.append(mask)
    }
    try Task.checkCancellation()
    guard id == revision else { throw CancellationError() }
    _ = tracker.update(
      zip(entities, masks).compactMap { entity, mask in
        mask.bounds.map { Detection(entity: entity, bounds: $0) }
      }, camera: frame.camera)
    selected = entities.first ?? tracker.tracks.first?.entity
    latest = SceneSelection(
      entities: entities, masks: masks, viewportToImage: transform, capturedAt: time)
    return latest
  }

  public func inspect(subject: String, count: Bool) async throws -> [Entity] {
    queriesInFlight += 1
    defer { queriesInFlight -= 1 }
    revision += 1
    let id = revision
    let transform = viewportToImage
    let time = capturedAt
    guard let frame = snapshot else { throw PerceptionError.insufficientDepth }
    if !count, let selected,
      subject == "subject" || selected.label.localizedCaseInsensitiveContains(subject)
    {
      return try await refreshSelection(isQuery: true).entities
    }
    let masks: [SubjectMask]
    if subject == "subject" && !count {
      masks = [try await segmenter.segment(frame.image, prompt: .point(.init(x: 0.5, y: 0.5)))]
    } else {
      masks = try await instances.instances(frame.image)
    }
    var found: [Entity] = []
    var accepted: [SubjectMask] = []
    for mask in masks {
      try Task.checkCancellation()
      let label = try await identifier.identify(frame.image, mask: mask)
      try Task.checkCancellation()
      // A semantic voice choice must match actual on-device evidence.
      let noun = subject.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
      let singular = noun.hasSuffix("s") ? String(noun.dropLast()) : noun
      guard noun == "subject" || (label?.label.lowercased().contains(singular) == true) else {
        continue
      }
      guard
        var entity = try? MaskLifter.lift(
          mask: mask, depth: frame.depth, camera: frame.camera,
          label: label?.label ?? "subject")
      else { continue }
      if let label {
        entity.attrs["identifiedOnDevice"] = .bool(true)
        entity.attrs["identificationConfidence"] = .number(label.confidence)
      }
      found.append(entity)
      accepted.append(mask)
      if !count { break }
    }
    try Task.checkCancellation()
    guard id == revision else { throw CancellationError() }
    latest = SceneSelection(
      entities: found, masks: accepted, viewportToImage: transform, capturedAt: time)
    selected = found.first
    tracker = EntityTracker()
    _ = tracker.update(
      zip(found, accepted).compactMap { entity, mask in
        mask.bounds.map { Detection(entity: entity, bounds: $0) }
      }, camera: frame.camera)
    return found
  }
}
