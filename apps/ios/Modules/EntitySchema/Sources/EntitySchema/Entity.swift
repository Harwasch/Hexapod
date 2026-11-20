import Foundation

/// JSON data carried by module-defined attributes, without an untyped Any boundary.
public enum JSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case integer(Int64)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    // JSON has one numeric domain; an integral Double may be encoded without a
    // decimal point. Compare it exactly to integers without rounding large IDs.
    public static func == (lhs: JSONValue, rhs: JSONValue) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null): return true
        case (.bool(let a), .bool(let b)): return a == b
        case (.integer(let a), .integer(let b)): return a == b
        case (.number(let a), .number(let b)): return a == b
        case (.integer(let a), .number(let b)), (.number(let b), .integer(let a)):
            return Int64(exactly: b) == a
        case (.string(let a), .string(let b)): return a == b
        case (.array(let a), .array(let b)): return a == b
        case (.object(let a), .object(let b)): return a == b
        default: return false
        }
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Int64.self) { self = .integer(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else { self = .object(try container.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

public struct Vector3: Codable, Equatable, Sendable {
    public var x: Double
    public var y: Double
    public var z: Double

    public init(x: Double, y: Double, z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }
}

/// WGS84 degrees and altitude in metres above the ellipsoid.
public struct GeoCoordinate: Codable, Equatable, Sendable {
    public var lat: Double
    public var lon: Double
    public var alt: Double

    public init(lat: Double, lon: Double, alt: Double) {
        self.lat = lat
        self.lon = lon
        self.alt = alt
    }
}

public struct Anchor: Codable, Equatable, Sendable {
    /// ARKit session world coordinates in metres, right-handed with Y up.
    public var world: Vector3
    public var geo: GeoCoordinate?

    public init(world: Vector3, geo: GeoCoordinate? = nil) {
        self.world = world
        self.geo = geo
    }
}

public struct MaskReference: Codable, Equatable, Sendable {
    /// Keyframe identifier and bundle-relative mask path; no inline image data.
    public var keyframe: String
    public var ref: String

    public init(keyframe: String, ref: String) {
        self.keyframe = keyframe
        self.ref = ref
    }
}

public struct EntitySource: Codable, Equatable, Sendable {
    public var module: String
    public var model: String

    public init(module: String, model: String) {
        self.module = module
        self.model = model
    }
}

/// Transport contract. Validate against entity.schema.json at ingestion boundaries.
public struct Entity: Codable, Equatable, Sendable {
    public var label: String
    public var anchor: Anchor
    /// Axis-aligned dimensions in metres in the anchor's world coordinate frame.
    public var extent: Vector3
    public var masks: [MaskReference]
    public var confidence: Double
    public var attrs: [String: JSONValue]
    public var source: EntitySource

    public init(
        label: String, anchor: Anchor, extent: Vector3, masks: [MaskReference],
        confidence: Double, attrs: [String: JSONValue], source: EntitySource
    ) {
        self.label = label
        self.anchor = anchor
        self.extent = extent
        self.masks = masks
        self.confidence = confidence
        self.attrs = attrs
        self.source = source
    }
}
