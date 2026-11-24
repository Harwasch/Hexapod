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
  /// Perception already separates instances. Nearby subjects must retain separate pins.
  public mutating func replaceSelection(_ entities: [Entity]) {
    reset()
    self.entities = entities
  }
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
    project(
      width: width, height: height, pixels: pixels, viewportToImage: viewportToImage,
      outputWidth: outputSize, outputHeight: outputSize)
  }

  public static func project(
    width: Int, height: Int, pixels: [UInt8], viewportToImage: [Double],
    outputWidth: Int, outputHeight: Int
  ) -> Self? {
    guard width > 0, height > 0, pixels.count / width == height, pixels.count % width == 0,
      viewportToImage.count == 6, viewportToImage.allSatisfy(\.isFinite),
      outputWidth > 0, outputHeight > 0, outputWidth <= 2048, outputHeight <= 2048
    else { return nil }
    let mapping = ViewportMapping(viewportToImage: viewportToImage)
    var output = [UInt8](repeating: 0, count: outputWidth * outputHeight)
    for y in 0..<outputHeight {
      for x in 0..<outputWidth {
        guard
          let p = mapping.imagePoint(
            x: (Double(x) + 0.5) / Double(outputWidth),
            y: (Double(y) + 0.5) / Double(outputHeight)),
          p.x >= 0, p.x < 1, p.y >= 0, p.y < 1
        else { continue }
        // Pixel centres, not pixel corners. Clamp the bilinear footprint at image edges.
        let sx = max(0, min(Double(width - 1), p.x * Double(width) - 0.5))
        let sy = max(0, min(Double(height - 1), p.y * Double(height) - 0.5))
        let x0 = Int(sx)
        let y0 = Int(sy)
        let x1 = min(width - 1, x0 + 1)
        let y1 = min(height - 1, y0 + 1)
        let fx = sx - Double(x0)
        let fy = sy - Double(y0)
        let top = Double(pixels[y0 * width + x0]) * (1 - fx) + Double(pixels[y0 * width + x1]) * fx
        let bottom =
          Double(pixels[y1 * width + x0]) * (1 - fx) + Double(pixels[y1 * width + x1]) * fx
        output[y * outputWidth + x] = UInt8((top * (1 - fy) + bottom * fy).rounded())
      }
    }
    return Self(width: outputWidth, height: outputHeight, pixels: output)
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
