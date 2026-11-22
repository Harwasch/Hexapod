import EntitySchema
import Foundation
import Router

public struct AnswerPresentation: Sendable {
  public private(set) var entities: [Entity] = []
  public private(set) var speech = ""
  public private(set) var isComplete = false
  public private(set) var isUnavailable = false
  public private(set) var isConnecting = false
  public init() {}
  public mutating func reset() { self = Self() }
  public mutating func receive(_ event: AskEvent) {
    isConnecting = event == .status("Connecting…")
    switch event {
    case .entity(let entity):
      if let index = entities.firstIndex(where: { EntityMatch.matches($0, entity) }) {
        entities[index] = entity
      } else {
        entities.append(entity)
      }
    case .speech(let text): speech = text
    case .done: isComplete = true
    case .status(let text):
      if ["Fast answer unavailable", "Deep answer unavailable", "Answer interrupted"].contains(text)
      {
        isUnavailable = true
      }
    }
  }
}

/// A viewport-oriented union of subject masks. No subject means no dimming.
/// The App applies ARFrame.displayTransform before submitting pixels.
public struct HighlightMask: Sendable {
  public var width: Int
  public var height: Int
  public var pixels: [UInt8]
  public init(width: Int, height: Int, pixels: [UInt8]) {
    self.width = width
    self.height = height
    self.pixels = pixels
  }
  public var isValid: Bool {
    width > 0 && height > 0 && pixels.count / width == height && pixels.count % width == 0
  }

  /// Inverse affine transform maps normalized viewport UV into normalized camera UV.
  public static func project(
    width: Int, height: Int, pixels: [UInt8],
    viewportToImage: [Double], outputSize: Int = 256
  ) -> Self? {
    guard width > 0, height > 0, pixels.count / width == height, pixels.count % width == 0,
      viewportToImage.count == 6, viewportToImage.allSatisfy(\.isFinite),
      outputSize > 0 && outputSize <= 2048
    else { return nil }
    let t = viewportToImage
    var output = [UInt8](repeating: 0, count: outputSize * outputSize)
    for y in 0..<outputSize {
      for x in 0..<outputSize {
        let u = (Double(x) + 0.5) / Double(outputSize)
        let v = (Double(y) + 0.5) / Double(outputSize)
        let ix = t[0] * u + t[2] * v + t[4]
        let iy = t[1] * u + t[3] * v + t[5]
        if ix >= 0 && ix < 1 && iy >= 0 && iy < 1 {
          output[y * outputSize + x] =
            pixels[Int(iy * Double(height)) * width + Int(ix * Double(width))]
        }
      }
    }
    return Self(width: outputSize, height: outputSize, pixels: output)
  }
}

/// Evidence-backed card facts. Unknown invasive status never becomes an Invasive chip.
public struct EntityCardFacts: Sendable {
  public let latinName: String?
  public let invasive: Bool
  public init(_ entity: Entity) {
    if case .string(let name) = entity.attrs["latinName"] {
      latinName = name
    } else if case .string(let name) = entity.attrs["scientificName"] {
      latinName = name
    } else {
      latinName = nil
    }
    if case .object(let lookup) = entity.attrs["invasive"] {
      invasive = lookup["listed"] == .bool(true)
    } else {
      invasive = entity.attrs["invasive"] == .bool(true)
    }
  }
}
