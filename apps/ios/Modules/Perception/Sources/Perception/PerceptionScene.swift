import EntitySchema
import Foundation

public struct SceneSelection: Sendable {
  public var entities: [Entity]
  public var surfaces: [SubjectSurface]
  public var masks: [SubjectMask]
  public var viewportToImage: [Double] = [1, 0, 0, 1, 0, 0]
  public var camera: CameraGeometry?
  public var capturedAt: TimeInterval = 0
  public init(
    entities: [Entity], masks: [SubjectMask], viewportToImage: [Double] = [1, 0, 0, 1, 0, 0],
    capturedAt: TimeInterval = 0, camera: CameraGeometry? = nil, surfaces: [SubjectSurface] = []
  ) {
    self.surfaces = surfaces
    self.entities = entities
    self.masks = masks
    self.viewportToImage = viewportToImage
    self.capturedAt = capturedAt
    self.camera = camera
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
  private func clearSelection() {
    selected = nil
    tracker = EntityTracker()
    latest = SceneSelection(
      entities: [], masks: [], viewportToImage: viewportToImage, capturedAt: capturedAt)
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
    let mask: SubjectMask
    do {
      mask = try await segmenter.segment(frame.image, prompt: .point(point))
    } catch PerceptionError.noSubject {
      guard id == revision else { throw CancellationError() }
      clearSelection()
      return latest
    }
    try Task.checkCancellation()
    guard id == revision else { throw CancellationError() }
    guard mask.bounds != nil else {
      clearSelection()
      return latest
    }
    let lifted = try MaskLifter.lift(
      mask: mask, depth: frame.depth, camera: frame.camera, label: "subject")
    // Same-frame masks are exact; after movement use the tracker's world-distance/overlap gate.
    if let index = tracker.tracks.indices.first(where: { index in
      if latest.capturedAt == time, latest.masks.count == latest.entities.count,
        latest.masks.indices.contains(index),
        (latest.masks[index].bounds?.iou(mask.bounds!) ?? 0) > 0.7
      {
        return true
      }
      let previous = tracker.tracks[index].entity
      let p = previous.anchor.world
      let q = lifted.anchor.world
      let distance = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2) + pow(p.z - q.z, 2))
      return distance <= tracker.maximumDistance * 0.5
        && (frame.camera.projectedBounds(previous)?.iou(mask.bounds!) ?? 0) >= 0.05
    }) {
      var remaining = tracker.tracks.map(\.entity)
      remaining.remove(at: index)
      selected = remaining.last
      var masks =
        latest.capturedAt == time && latest.masks.count == latest.entities.count ? latest.masks : []
      if masks.indices.contains(index) { masks.remove(at: index) }
      latest = SceneSelection(
        entities: remaining, masks: masks, viewportToImage: transform, capturedAt: time,
        camera: frame.camera)
      rebuildTracks(remaining, camera: frame.camera)
      return latest
    }
    let label = try await identifier.identify(frame.image, mask: mask)
    try Task.checkCancellation()
    guard id == revision else { throw CancellationError() }
    var entity = lifted
    entity.label = label?.label ?? "subject"
    if let label {
      entity.attrs["identifiedOnDevice"] = .bool(true)
      entity.attrs["identificationConfidence"] = .number(label.confidence)
    }
    // Preserve the selected set and its order. Old masks must never be mixed with a new camera.
    var entities = tracker.tracks.isEmpty ? latest.entities : tracker.tracks.map(\.entity)
    entities.append(entity)
    selected = entity
    var masks =
      latest.capturedAt == time && latest.masks.count == latest.entities.count ? latest.masks : []
    masks.append(mask)
    latest = SceneSelection(
      entities: entities, masks: masks, viewportToImage: transform, capturedAt: time,
      camera: frame.camera)
    rebuildTracks(entities, camera: frame.camera)
    return latest
  }

  private func rebuildTracks(_ entities: [Entity], camera: CameraGeometry) {
    tracker = EntityTracker()
    _ = tracker.update(
      entities.compactMap { entity in
        camera.projectedBounds(entity).map { Detection(entity: entity, bounds: $0) }
      }, camera: camera)
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
      entities: tracker.tracks.map(\.entity), masks: masks, viewportToImage: transform,
      capturedAt: time, camera: frame.camera)
    return latest
  }

  /// One foreground-instance request for the entire frame, shared by selection and refresh.
  /// No repeated Vision request per subject, and classification runs only for a new tap.
  public func foregroundSelection(tap: ImagePoint?) async throws -> SceneSelection {
    guard tap != nil || !latest.entities.isEmpty else { return latest }
    guard queriesInFlight == 0 else { throw CancellationError() }
    queriesInFlight += 1
    defer { queriesInFlight -= 1 }
    if tap != nil { revision += 1 }
    let id = revision
    guard let frame = snapshot else { throw PerceptionError.insufficientDepth }
    let transform = viewportToImage
    let time = capturedAt
    let masks = try await instances.instances(frame.image)
    try Task.checkCancellation()
    guard id == revision else { throw CancellationError() }
    func contains(_ mask: SubjectMask, _ point: ImagePoint) -> Bool {
      guard mask.isValid, point.x.isFinite, point.y.isFinite,
        (0..<1).contains(point.x), (0..<1).contains(point.y)
      else { return false }
      let x = Int(point.x * Double(mask.width))
      let y = Int(point.y * Double(mask.height))
      return mask.pixels[y * mask.width + x] > 0
    }
    var entities = latest.entities
    var choices = entities.map { previous -> Int? in
      if let point = frame.camera.project(previous.anchor.world),
        let index = masks.firstIndex(where: { contains($0, point) })
      {
        return index
      }
      // Handles rings/handles and other subjects whose median anchor lies outside their mask.
      guard
        let surface = latest.surfaces.first(where: {
          $0.entity.label == previous.label && $0.entity.anchor.world == previous.anchor.world
        })
      else { return nil }
      let points = surface.mesh.points
      let step = max(
        1,
        (points.count + PerceptionLimits.maximumAssociationPoints - 1)
          / PerceptionLimits.maximumAssociationPoints)
      let projected = stride(from: 0, to: points.count, by: step).compactMap {
        frame.camera.project(points[$0])
      }
      let votes = masks.map { mask in projected.filter { contains(mask, $0) }.count }
      guard let maximum = votes.max(), maximum > 0 else { return nil }
      return votes.firstIndex(of: maximum)
    }
    if let tap {
      guard let choice = masks.firstIndex(where: { contains($0, tap) }) else {
        clearSelection()
        return latest
      }
      if let existing = choices.firstIndex(where: { $0 == choice }) {
        entities.remove(at: existing)
        choices.remove(at: existing)
      } else {
        let label = try await identifier.identify(frame.image, mask: masks[choice])
        try Task.checkCancellation()
        guard id == revision else { throw CancellationError() }
        var entity = try MaskLifter.lift(
          mask: masks[choice], depth: frame.depth,
          camera: frame.camera, label: label?.label ?? "subject")
        if let label {
          entity.attrs["identifiedOnDevice"] = .bool(true)
          entity.attrs["identificationConfidence"] = .number(label.confidence)
        }
        entities.append(entity)
        choices.append(choice)
      }
    }
    var surfaces: [SubjectSurface] = []
    var accepted: [SubjectMask] = []
    for index in entities.indices {
      let previous = entities[index]
      if let choice = choices[index],
        var entity = try? MaskLifter.lift(
          mask: masks[choice], depth: frame.depth,
          camera: frame.camera, label: previous.label),
        let mesh = try? DepthSurface.lift(
          mask: masks[choice], depth: frame.depth, camera: frame.camera)
      {
        entity.attrs.merge(previous.attrs) { current, _ in current }
        entities[index] = entity
        accepted.append(masks[choice])
        surfaces.append(SubjectSurface(entity: entity, mesh: mesh, capturedAt: time))
      } else if let old = latest.surfaces.first(where: {
        $0.entity.label == previous.label && $0.entity.anchor.world == previous.anchor.world
      }) {
        // Keep the last world geometry until a refresh succeeds or selection removes it.
        surfaces.append(old)
      }
    }
    try Task.checkCancellation()
    guard id == revision else { throw CancellationError() }
    selected = entities.last
    rebuildTracks(entities, camera: frame.camera)
    latest = SceneSelection(
      entities: entities, masks: accepted, viewportToImage: transform,
      capturedAt: time, camera: frame.camera, surfaces: surfaces)
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
      entities: found, masks: accepted, viewportToImage: transform, capturedAt: time,
      camera: frame.camera)
    selected = found.first
    tracker = EntityTracker()
    _ = tracker.update(
      zip(found, accepted).compactMap { entity, mask in
        mask.bounds.map { Detection(entity: entity, bounds: $0) }
      }, camera: frame.camera)
    return found
  }
}
