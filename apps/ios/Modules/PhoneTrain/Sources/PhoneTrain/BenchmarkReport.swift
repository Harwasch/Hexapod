import Foundation
import PhoneTrainPrep

public struct BenchmarkCase: Codable, Sendable, Equatable {
  public let maxGaussians: Int
  public let maxSide: Int
  /// Six runs with identical iteration/SH schedules and 1/8 holdout policy.
  public static let sweep = [200_000, 400_000, 800_000].flatMap { count in
    [640, 960].map { Self(maxGaussians: count, maxSide: $0) }
  }
}

public struct BenchmarkRun: Codable, Sendable {
  public let configuration: BenchmarkCase
  public let training: PhoneTrainingConfig
  public init(configuration: BenchmarkCase) {
    self.configuration = configuration
    training = .benchmark(maxGaussians: configuration.maxGaussians)
  }
  public var status = "preparing"
  public var outputPath: String?
  public var error: String?
  public var preparationSeconds: Double?
  public var trainingSeconds: Double?
  public var evaluationSeconds: Double?
  public var totalSeconds: Double?
  public var estimatedPeakBytes: Int64?
  /// Sampled process peak, not a claim to capture between-sample allocation peaks.
  public var peakPhysicalFootprint: UInt64?
  public var availableAtStart: UInt64?
  public var thermalAtStart: Int?
  public var thermalAtEnd: Int?
  public var psnr: Float?
  public var ssim: Float?
  public var evaluatedViews: Int?
  public var seedTargetReached: Bool?
  public var seedCount: Int?
  public var keyframeCount: Int?
  public var voxelSize: Double?
  public var samples: [TrainSample] = []
  public var worstThermal: Int {
    ([thermalAtStart, thermalAtEnd].compactMap { $0 } + samples.map(\.thermalState)).max() ?? 0
  }
}

public struct BenchmarkReport: Codable, Sendable {
  public var schemaVersion = 1
  public let capturePath: String
  public let startedAt: Date
  public let device: String
  public let operatingSystem: String
  public let physicalMemory: UInt64
  public var prep: PrepConfig
  public var thermalTimeline: [ThermalSample] = []
  public var stoppedForCriticalThermal = false
  public var stopReason: String?
  public var runs: [BenchmarkRun] = []

  public func write(to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    try encoder.encode(self).write(to: url, options: .atomic)
  }
}

public struct ThermalSample: Codable, Sendable {
  public let elapsedSeconds: Double
  public let state: Int
  public let physicalFootprint: UInt64?
  public let availableMemory: UInt64
}
