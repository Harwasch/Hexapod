import Foundation

public enum ScanTip: String, Sendable {
  case around = "Around"
  case higher = "Higher"
  case slower = "Slower"
  case closer = "Closer"
  public var symbol: String {
    switch self {
    case .around: "figure.walk.circle"
    case .higher: "arrow.up"
    case .slower: "tortoise.fill"
    case .closer: "arrow.down.right.and.arrow.up.left"
    }
  }
}

/// A small summary of observed geometry plus the camera pose, never geometry readback.
public struct ScanGuidanceSample: Sendable {
  public var center: SIMD3<Float>
  public var camera: SIMD3<Float>
  public var observedCells: Int
  public var shortSpreadFraction: Float
  public var timestamp: Double
  public init(
    center: SIMD3<Float>, camera: SIMD3<Float>, observedCells: Int,
    shortSpreadFraction: Float, timestamp: Double
  ) {
    self.center = center
    self.camera = camera
    self.observedCells = observedCells
    self.shortSpreadFraction = shortSpreadFraction
    self.timestamp = timestamp
  }
}

/// Fixed-size pose history and Schmitt thresholds. No Apple frameworks or timers.
/// Elevation bands match quality.py (-10°, 20°, 45°); the largest empty arc
/// uses actual extrema within 15° bins rather than counting occupied sectors.
public struct ScanGuidance: Sendable {
  public enum Thresholds {
    public static let farEnter: Float = 3.5, farExit: Float = 3.0
    public static let gapEnter: Float = 60, gapExit: Float = 45
    public static let raisedEnter: Float = 0.20, raisedExit: Float = 0.25
    public static let spreadEnter: Float = 0.30, spreadExit: Float = 0.15
    public static let minimumDwell = 2.0, candidateDwell = 0.6
    public static let sectorSeconds = 0.4, sampleInterval = 0.09
    public static let blurEnter = 0.5, blurExit = 0.1
  }
  public private(set) var tip: ScanTip?
  public private(set) var completedSectors: UInt32 = 0
  public var fraction: Double {
    Double(completedSectors.nonzeroBitCount) / Double(CoverageHash.sectorCount)
  }
  public private(set) var largestGap: Float = 360
  private var anchor: SIMD3<Float>?
  private var minimum = Array(repeating: Float.infinity, count: CoverageHash.sectorCount)
  private var maximum = Array(repeating: -Float.infinity, count: CoverageHash.sectorCount)
  private var sectorTime = Array(repeating: 0.0, count: CoverageHash.sectorCount)
  private var poses = 0, raised = 0, above = 0
  private var lastSample = -Double.infinity
  private var lastUpdate = -Double.infinity
  private var changedAt = -Double.infinity
  private var candidate: ScanTip?
  private var candidateAt = -Double.infinity
  private var far = false, blurred = false, needsAround = true, needsHigher = true
  public init() {}

  /// Blur is 0 for steady, 1 for slowDown, 2 for tooFast from Capture's meter.
  /// Returns newly completed sector bits; the UI emits at most one subtle haptic per poll.
  @discardableResult public mutating func update(
    _ sample: ScanGuidanceSample?, blur: Double,
    now: Double
  ) -> UInt32 {
    guard now.isFinite, now >= lastUpdate else { return 0 }
    lastUpdate = now
    blurred = blur >= (blurred ? Thresholds.blurExit : Thresholds.blurEnter)
    var newSectors: UInt32 = 0
    if let s = sample, s.timestamp.isFinite, s.timestamp - lastSample >= Thresholds.sampleInterval,
      s.observedCells > 0, Self.finite(s.center), Self.finite(s.camera)
    {
      if anchor == nil { anchor = s.center }
      let offset = s.camera - anchor!
      let distance = sqrt(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z)
      if distance > 0.1 {
        far = distance > (far ? Thresholds.farExit : Thresholds.farEnter)
        let dt = lastSample.isFinite ? min(0.2, s.timestamp - lastSample) : 0
        // Blurred or distant views do not complete sectors or improve elevation credit.
        if !blurred && !far {
          let angle = (atan2(offset.z, offset.x) * 180 / .pi + 360).truncatingRemainder(
            dividingBy: 360)
          let sector = min(CoverageHash.sectorCount - 1, Int(angle / 15))
          minimum[sector] = min(minimum[sector], angle)
          maximum[sector] = max(maximum[sector], angle)
          sectorTime[sector] += dt
          if sectorTime[sector] >= Thresholds.sectorSeconds {
            let bit: UInt32 = 1 << sector
            newSectors = bit & ~completedSectors
            completedSectors |= bit
          }
          poses += 1
          let elevation = asin(min(1, max(-1, offset.y / distance))) * 180 / .pi
          if elevation >= 20 { raised += 1 }
          if elevation >= 45 { above += 1 }
          var first: Float?
          var previous: Float?
          var gap: Float = 0
          for i in minimum.indices where minimum[i].isFinite {
            if first == nil { first = minimum[i] }
            if let previous { gap = max(gap, minimum[i] - previous) }
            gap = max(gap, maximum[i] - minimum[i])
            previous = maximum[i]
          }
          if let first, let previous { gap = max(gap, first + 360 - previous) }
          largestGap = gap
          needsAround = gap >= (needsAround ? Thresholds.gapExit : Thresholds.gapEnter)
          let raisedShare = Float(raised) / Float(poses)
          needsHigher =
            above == 0
            && raisedShare < (needsHigher ? Thresholds.raisedExit : Thresholds.raisedEnter)
        }
        lastSample = s.timestamp
      }
    }
    let desired: ScanTip?
    if blurred {
      desired = .slower
    } else if far {
      desired = .closer
    } else if anchor == nil {
      desired = nil
    } else if needsAround {
      desired = .around
    } else if needsHigher {
      desired = .higher
    } else if let sample,
      sample.shortSpreadFraction
        >= (tip == .around ? Thresholds.spreadExit : Thresholds.spreadEnter)
    {
      desired = .around  // More angles on the surfaces still short of spread.
    } else {
      desired = nil
    }
    if candidate != desired {
      candidate = desired
      candidateAt = now
    }
    if desired != tip, now - candidateAt >= Thresholds.candidateDwell,
      now - changedAt >= Thresholds.minimumDwell
    {
      tip = desired
      changedAt = now
    }
    return newSectors
  }
  private static func finite(_ v: SIMD3<Float>) -> Bool {
    v.x.isFinite && v.y.isFinite && v.z.isFinite
  }
}
