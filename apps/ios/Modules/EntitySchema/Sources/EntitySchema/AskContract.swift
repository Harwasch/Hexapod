import Foundation

/// One keyframe the asker points at, by reference into its capture bundle; no inline pixels.
public struct KeyframeRef: Codable, Equatable, Sendable {
    /// Frame index in the bundle's `frames.jsonl`.
    public var index: Int
    /// Bundle-relative image reference (a decoded frame or an uploaded still).
    public var image: String
    /// Bundle-relative float16 depth file, when the frame has LiDAR depth.
    public var depth: String?
    /// Pose reference, `frames.jsonl#<index>`: transform and intrinsics live in the frame log.
    public var pose: String

    public init(index: Int, image: String, depth: String? = nil, pose: String) {
        self.index = index
        self.image = image
        self.depth = depth
        self.pose = pose
    }
}

/// A spoken question plus what the phone already knows about the scene.
public struct AskRequest: Codable, Equatable, Sendable {
    /// Client-chosen identifier, so a retried request is the same question.
    public var id: String
    public var question: String
    public var scanId: String?
    public var location: GeoCoordinate?
    public var keyframes: [KeyframeRef]
    /// Entities the phone's own perception already found.
    public var deviceEntities: [Entity]

    public init(
        id: String, question: String, scanId: String? = nil, location: GeoCoordinate? = nil,
        keyframes: [KeyframeRef] = [], deviceEntities: [Entity] = []
    ) {
        self.id = id
        self.question = question
        self.scanId = scanId
        self.location = location
        self.keyframes = keyframes
        self.deviceEntities = deviceEntities
    }
}

/// One step of an answer stream. On the wire: `{"type": "entity" | "speech" | "status" | "done", ...}`,
/// with `entity` carrying an Entity and `speech` / `status` carrying `text`.
public enum AskEvent: Codable, Equatable, Sendable {
    case entity(Entity)
    case speech(String)
    case status(String)
    case done

    private enum CodingKeys: String, CodingKey { case type, entity, text }
    private enum Kind: String, Codable { case entity, speech, status, done }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .type) {
        case .entity: self = .entity(try container.decode(Entity.self, forKey: .entity))
        case .speech: self = .speech(try container.decode(String.self, forKey: .text))
        case .status: self = .status(try container.decode(String.self, forKey: .text))
        case .done: self = .done
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .entity(let entity):
            try container.encode(Kind.entity, forKey: .type)
            try container.encode(entity, forKey: .entity)
        case .speech(let text):
            try container.encode(Kind.speech, forKey: .type)
            try container.encode(text, forKey: .text)
        case .status(let text):
            try container.encode(Kind.status, forKey: .type)
            try container.encode(text, forKey: .text)
        case .done:
            try container.encode(Kind.done, forKey: .type)
        }
    }
}

/// The slow path: anything that answers an AskRequest as a stream (the cloud reasoner, a test double).
public protocol DeepAnswering: Sendable {
    func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error>
}
