import Foundation

public struct PreparedFrame: Codable, Sendable {
  public let source: CaptureFrame
  public let pose: [Double]
  public let blurScore: Double
  public let isHoldout: Bool
}

public enum Keyframes {
  /// Rec. 709 luma in [0,1] from packed 8-bit RGB. A plain loop with typed terms: the
  /// closure form timed out Xcode's type checker.
  public static func luma(rgb: [UInt8]) -> [Double] {
    let red: Double = 0.2126 / 255
    let green: Double = 0.7152 / 255
    let blue: Double = 0.0722 / 255
    var result = [Double]()
    result.reserveCapacity(rgb.count / 3)
    var index = 0
    while index + 2 < rgb.count {
      let r: Double = Double(rgb[index]) * red
      let g: Double = Double(rgb[index + 1]) * green
      let b: Double = Double(rgb[index + 2]) * blue
      result.append(r + g + b)
      index += 3
    }
    return result
  }

  /// Variance of the four-neighbour Laplacian. Input is downscaled luma [0,1].
  public static func blurScore(luma: [Double], width: Int, height: Int) throws -> Double {
    guard width >= 3, height >= 3, luma.count == width * height,
      luma.allSatisfy(\.isFinite)
    else { throw PrepError.invalid("Invalid luma raster") }
    var sum = 0.0
    var squares = 0.0
    var count = 0.0
    for y in 1..<(height - 1) {
      for x in 1..<(width - 1) {
        let i = y * width + x
        let value = luma[i - 1] + luma[i + 1] + luma[i - width] + luma[i + width] - 4 * luma[i]
        sum += value
        squares += value * value
        count += 1
      }
    }
    return max(0, squares / count - pow(sum / count, 2))
  }

  /// Scores must use one common raster scale. A global median makes the blur
  /// gate relative to this scan; relaxation never readmits blurry/limited views.
  public static func select(
    frames: [CaptureFrame], scores: [Int: Double], config: PrepConfig = .init()
  ) throws -> [PreparedFrame] {
    try config.validate()
    let normal = try frames.filter { $0.trackingState == "normal" }.map { frame in
      guard let score = scores[frame.index], score.isFinite, score >= 0 else {
        throw PrepError.invalid("Missing blur score for frame \(frame.index)")
      }
      return (frame, try Pose.corrected(frame.transform, config: config), score)
    }
    guard !normal.isEmpty else { throw PrepError.invalid("No normally tracked frames") }
    let orderedScores = normal.map { $0.2 }.sorted()
    let threshold = orderedScores[orderedScores.count / 2] * config.blurMedianFraction
    let sharp = normal.filter { $0.2 >= threshold }
    func pick(translation: Double, rotation: Double) -> [(CaptureFrame, [Double], Double)] {
      var kept: [(CaptureFrame, [Double], Double)] = []
      for candidate in sharp {
        if let last = kept.last {
          let delta = Pose.distance(last.1, candidate.1)
          if delta.metres < translation && delta.degrees < rotation { continue }
        }
        kept.append(candidate)
      }
      return kept
    }
    var kept = pick(translation: config.keyframeTranslation, rotation: config.keyframeRotation)
    if kept.count < config.minimumKeyframes {
      kept = pick(
        translation: min(config.keyframeTranslation, config.relaxedTranslation),
        rotation: min(config.keyframeRotation, config.relaxedRotation))
    }
    if kept.count > config.maximumKeyframes {
      let original = kept
      kept = (0..<config.maximumKeyframes).map {
        original[$0 * (original.count - 1) / (config.maximumKeyframes - 1)]
      }
    }
    guard kept.count >= 2 else {
      throw PrepError.invalid("At least two sharp keyframes are needed")
    }
    return kept.enumerated().map {
      PreparedFrame(
        source: $0.element.0, pose: $0.element.1, blurScore: $0.element.2,
        isHoldout: isHoldout($0.offset, every: config.holdoutEvery))
    }
  }

  /// msplat InputData::splitTrainTestIndices holds out indices 0, 8, 16, … .
  public static func isHoldout(_ index: Int, every: Int = 8) -> Bool { index % every == 0 }
}
