import EntitySchema
import Foundation

public enum CloudError: Error, Equatable, Sendable {
    case invalidUTF8
    case eventTooLarge
    case incompleteStream
    case server(Int)
    case invalidResponse
    case unavailablePlatform
}

/// Incremental UTF-8 SSE framing. Handles CR, LF, CRLF, comments and split code points.
public struct SSEParser: Sendable {
    private var line = Data()
    private var dataLines: [String] = []
    private var eventBytes = 0
    private var skipLF = false
    private var firstLine = true
    public let maximumEventBytes: Int

    public init(maximumEventBytes: Int = 1_048_576) {
        self.maximumEventBytes = maximumEventBytes
    }

    public mutating func feed(_ bytes: Data) throws -> [AskEvent] {
        var events: [AskEvent] = []
        for byte in bytes {
            if skipLF {
                skipLF = false
                if byte == 10 { continue }
            }
            if byte == 10 || byte == 13 {
                if let event = try consumeLine() { events.append(event) }
                skipLF = byte == 13
            } else {
                line.append(byte)
                if line.count + eventBytes > maximumEventBytes { throw CloudError.eventTooLarge }
            }
        }
        return events
    }

    private mutating func consumeLine() throws -> AskEvent? {
        guard var text = String(data: line, encoding: .utf8) else { throw CloudError.invalidUTF8 }
        line.removeAll(keepingCapacity: true)
        if firstLine {
            firstLine = false
            if text.hasPrefix("\u{FEFF}") { text.removeFirst() }
        }
        if text.isEmpty {
            guard !dataLines.isEmpty else { return nil }
            let json = dataLines.joined(separator: "\n")
            dataLines.removeAll(keepingCapacity: true)
            eventBytes = 0
            return try JSONDecoder().decode(AskEvent.self, from: Data(json.utf8))
        }
        if text.hasPrefix(":") { return nil }
        let parts = text.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        if parts[0] == "data" {
            var value = parts.count == 2 ? String(parts[1]) : ""
            if value.hasPrefix(" ") { value.removeFirst() }
            eventBytes += value.utf8.count + 1
            if eventBytes > maximumEventBytes { throw CloudError.eventTooLarge }
            dataLines.append(value)
        }
        return nil
    }
}
