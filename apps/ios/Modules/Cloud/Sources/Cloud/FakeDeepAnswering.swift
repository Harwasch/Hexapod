import EntitySchema
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Replays SSE bytes through the same parser and completion checks as the HTTP client.
/// The supplied recording is returned for every question; it is simulated evidence.
public struct FakeDeepAnswering: DeepAnswering {
    private let client: CloudAnswerer

    public init(recording: Data, chunkSize: Int = 7) {
        client = CloudAnswerer(
            baseURL: URL(string: "https://fixture.invalid")!, writeToken: "",
            transport: RecordedSSETransport(recording: recording, chunkSize: chunkSize),
            maximumRetries: 0
        )
    }

    /// Bundled synthetic yard-small height answer; no network or capture is needed.
    public static func yardSmall() throws -> Self {
        guard let url = Bundle.module.url(forResource: "yard-small-height", withExtension: "sse")
        else { throw CloudError.invalidResponse }
        return try Self(recording: Data(contentsOf: url))
    }

    public func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
        client.ask(request)
    }
}

public struct RecordedSSETransport: SSETransport {
    private let recording: Data
    private let chunkSize: Int

    public init(recording: Data, chunkSize: Int = 7) {
        self.recording = recording
        self.chunkSize = max(1, chunkSize)
    }

    public func connect(_ request: URLRequest) async throws -> SSEConnection {
        try Task.checkCancellation()
        let chunks = AsyncThrowingStream<Data, Error> { continuation in
            for offset in stride(from: 0, to: recording.count, by: chunkSize) {
                continuation.yield(recording.subdata(in: offset..<min(offset + chunkSize, recording.count)))
            }
            continuation.finish()
        }
        return SSEConnection(status: 200, chunks: chunks)
    }
}
