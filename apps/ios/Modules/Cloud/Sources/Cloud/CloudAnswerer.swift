import Networking
import EntitySchema
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct SSEConnection: Sendable {
    public var status: Int
    public var contentType: String
    public var retryAfter: Double?
    public var chunks: AsyncThrowingStream<Data, Error>
    public var cancel: @Sendable () -> Void

    public init(
        status: Int, contentType: String = "text/event-stream", retryAfter: Double? = nil,
        chunks: AsyncThrowingStream<Data, Error>, cancel: @escaping @Sendable () -> Void = {}
    ) {
        self.status = status
        self.contentType = contentType
        self.retryAfter = retryAfter
        self.chunks = chunks
        self.cancel = cancel
    }
}

public protocol SSETransport: Sendable {
    func connect(_ request: URLRequest) async throws -> SSEConnection
}

public struct URLSessionSSETransport: SSETransport {
    public init() {}

    public func connect(_ request: URLRequest) async throws -> SSEConnection {
        #if canImport(Darwin)
        // Apple URLSession streaming and radio cancellation require a device/Mac check.
        let (bytes, response) = try await APISession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse else { throw CloudError.invalidResponse }
        let (chunks, continuation) = AsyncThrowingStream<Data, Error>.makeStream()
        let task = Task {
            do {
                var buffer = Data()
                for try await byte in bytes {
                    try Task.checkCancellation()
                    buffer.append(byte)
                    if byte == 10 || byte == 13 || buffer.count >= 4096 {
                        continuation.yield(buffer)
                        buffer.removeAll(keepingCapacity: true)
                    }
                }
                if !buffer.isEmpty { continuation.yield(buffer) }
                continuation.finish()
            } catch { continuation.finish(throwing: error) }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return SSEConnection(
            status: http.statusCode, contentType: http.value(forHTTPHeaderField: "Content-Type") ?? "",
            retryAfter: http.value(forHTTPHeaderField: "Retry-After").flatMap(Double.init),
            chunks: chunks, cancel: { task.cancel() }
        )
        #else
        throw CloudError.unavailablePlatform
        #endif
    }
}

public struct CloudAnswerer: DeepAnswering {
    private let endpoint: URL
    private let writeToken: String
    private let transport: any SSETransport
    private let maximumRetries: Int
    private let retryDelay: Duration

    /// baseURL is the origin, or an API root ending in /api. Both serve /v1/ask.
    public init(
        baseURL: URL, writeToken: String, transport: any SSETransport = URLSessionSSETransport(),
        maximumRetries: Int = 2, retryDelay: Duration = .seconds(1)
    ) {
        self.endpoint = baseURL.appendingPathComponent("v1/ask")
        self.writeToken = writeToken
        self.transport = transport
        self.maximumRetries = min(RetryPolicy.maximumRetries, max(0, maximumRetries))
        self.retryDelay = retryDelay
    }

    public func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var urlRequest = URLRequest(url: endpoint)
                    urlRequest.httpMethod = "POST"
                    urlRequest.timeoutInterval = 90
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    urlRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    if !writeToken.isEmpty {
                        urlRequest.setValue("Bearer \(writeToken)", forHTTPHeaderField: "Authorization")
                    }
                    urlRequest.httpBody = try JSONEncoder().encode(request)
                    var delivered = false
                    for attempt in 0...maximumRetries {
                        try Task.checkCancellation()
                        continuation.yield(.status("Connecting…"))
                        var delay = retryDelay * (1 << attempt)
                        do {
                            let connection = try await transport.connect(urlRequest)
                            defer { connection.cancel() }
                            guard connection.status == 200 else {
                                if let seconds = connection.retryAfter, seconds.isFinite {
                                    delay = .seconds(min(30, max(0, seconds)))
                                }
                                throw CloudError.server(connection.status)
                            }
                            guard connection.contentType.lowercased().split(separator: ";").first?
                                .trimmingCharacters(in: .whitespaces) == "text/event-stream"
                            else { throw CloudError.invalidResponse }
                            var parser = SSEParser()
                            for try await chunk in connection.chunks {
                                try Task.checkCancellation()
                                for event in try parser.feed(chunk) {
                                    delivered = true
                                    continuation.yield(event)
                                    if event == .done {
                                        continuation.finish()
                                        return
                                    }
                                }
                            }
                            throw CloudError.incompleteStream
                        } catch {
                            try Task.checkCancellation()
                            guard !delivered, attempt < maximumRetries, Self.retryable(error) else { throw error }
                            try await Task.sleep(for: delay)
                        }
                    }
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private static func retryable(_ error: any Error) -> Bool {
        if case CloudError.server(let status) = error {
            return RetryPolicy.retryable(status: status)
        }
        if case CloudError.incompleteStream = error { return true }
        return RetryPolicy.retryable(error)
    }
}
