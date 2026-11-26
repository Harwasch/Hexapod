import Foundation
import PhoneTrainPrep

/// Initial measurement budgets, not measured duration promises. Quick targets
/// roughly two minutes; Best six to eight. Phone reports decide future defaults.
public struct PhoneTrainingConfig: Codable, Sendable, Equatable {
  public var iterations: Int32
  public var maxGaussians: Int
  public var shDegree: Int32
  public var numDownscales: Int32
  public var resolutionSchedule: Int32
  public var warmupLength: Int32
  public var gpuStepTimeoutSeconds: Double
  public var ssimWeight: Float
  public var seriousSleepMilliseconds: UInt64
  public var memoryBudgetFraction: Double
  public static let quick = Self(
    iterations: 1_500, maxGaussians: 300_000, shDegree: 1,
    numDownscales: 1, resolutionSchedule: 750, warmupLength: 100, gpuStepTimeoutSeconds: 60,
    ssimWeight: 0.2,
    seriousSleepMilliseconds: 100, memoryBudgetFraction: 0.7)
  public static let best = Self(
    iterations: 5_000, maxGaussians: 800_000, shDegree: 2,
    numDownscales: 2, resolutionSchedule: 1_500, warmupLength: 100, gpuStepTimeoutSeconds: 60,
    ssimWeight: 0.2,
    seriousSleepMilliseconds: 100, memoryBudgetFraction: 0.7)

  public func validate() throws {
    guard (2...1_000_000).contains(iterations), (0...4).contains(shDegree),
      iterations > shDegree, (0...4).contains(numDownscales),
      resolutionSchedule > (numDownscales > 0 ? 1 : 0),
      Int64(numDownscales) * Int64(resolutionSchedule) < Int64(iterations),
      (2...Int(Int32.max)).contains(maxGaussians), warmupLength >= 0,
      gpuStepTimeoutSeconds.isFinite, gpuStepTimeoutSeconds > 0,
      (0...1).contains(ssimWeight), memoryBudgetFraction.isFinite,
      memoryBudgetFraction > 0, memoryBudgetFraction <= 1,
      seriousSleepMilliseconds <= 60_000
    else { throw PrepError.invalid("Invalid training policy") }
  }

  /// Resolution stages as msplat's fixed-interval scheduler sees them: the factor at
  /// iteration i is 2^(numDownscales - i / S), with S the second stage's first iteration
  /// (TrainingPlan.swift nativeDownscaleFactor), so every later stage must start on a
  /// multiple of S. Call only after `validate()`.
  public func resolutionStages() -> [(iterations: ClosedRange<Int32>, downscaleFactor: Int32)] {
    (0...Int(numDownscales)).map { index in
      let level = Int32(index)
      let start = index == 0 ? 1 : level * resolutionSchedule
      let end =
        index == Int(numDownscales) ? iterations : (level + 1) * resolutionSchedule - 1
      return (start...end, Int32(1) << (Int(numDownscales) - index))
    }
  }

  public static func benchmark(maxGaussians: Int) -> Self {
    var config = quick
    config.iterations = 600
    config.maxGaussians = maxGaussians
    config.resolutionSchedule = 300
    return config
  }
}

public struct TrainSample: Codable, Sendable {
  public let step: Int
  public let wallSeconds: Double
  public let elapsedSeconds: Double
  public let completedMilliseconds: Float
  public let gpuMilliseconds: Float?
  public let loss: Float?
  /// Held-out PSNR is only available after evaluate(); never derive it from loss.
  public let psnr: Float?
  public let splatCount: Int
  public let etaSeconds: Double
  public let physicalFootprint: UInt64?
  public let availableMemory: UInt64
  public let thermalState: Int
}

public struct TrainProgress: Sendable {
  public let sample: TrainSample?
  public let message: String
  public let psnr: Float?
}

public struct TrainingCheckpoint: Codable, Sendable {
  public let capturePath: String
  public let prep: PrepConfig
  public let training: PhoneTrainingConfig
  public let step: Int
}
