import Cloud
import EntitySchema
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing

private func recording(_ name: String) throws -> Data {
    let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    return try Data(contentsOf: root.appendingPathComponent("Sources/Cloud/Resources/\(name).sse"))
}

@Test func recordedYardStreamsMatchServerAndSurviveEveryByteSplit() async throws {
    let repo = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
    for name in ["yard-small-height", "yard-small-count"] {
        let data = try recording(name)
        #expect(data == (try Data(contentsOf: repo.appendingPathComponent("apps/api/app/fixtures/ask/\(name).sse"))))
        var expectedParser = SSEParser()
        let expected = try expectedParser.feed(data)
        #expect(expected.count == 4)
        #expect(expected.last == .done)
        for size in [1, 7, data.count] {
            let client = CloudAnswerer(
                baseURL: URL(string: "https://fixture.invalid")!, writeToken: "",
                transport: RecordedSSETransport(recording: data, chunkSize: size)
            )
            var events: [AskEvent] = []
            for try await event in client.ask(AskRequest(id: "recorded", question: "Replay")) {
                events.append(event)
            }
            #expect(events == [.status("Connecting…")] + expected)
        }
        var fakeEvents: [AskEvent] = []
        for try await event in FakeDeepAnswering(recording: data).ask(AskRequest(id: "fake", question: "Replay")) {
            fakeEvents.append(event)
        }
        #expect(fakeEvents == [.status("Connecting…")] + expected)
    }
}

@Test func bundledFakeAndTruncatedRecording() async throws {
    var events: [AskEvent] = []
    let fake = try FakeDeepAnswering.yardSmall()
    for try await event in fake.ask(AskRequest(id: "bundled", question: "Height")) {
        events.append(event)
    }
    #expect(events.last == .done)
    let partial = Data("data: {\"type\":\"status\",\"text\":\"Simulated\"}\n\n".utf8)
    do {
        for try await _ in FakeDeepAnswering(recording: partial).ask(AskRequest(id: "partial", question: "Replay")) {}
        Issue.record("Expected incomplete stream")
    } catch { #expect(error as? CloudError == .incompleteStream) }
}

private func stream(_ value: String) -> AsyncThrowingStream<Data, Error> {
    AsyncThrowingStream { continuation in
        continuation.yield(Data(value.utf8))
        continuation.finish()
    }
}

private actor Transport: SSETransport {
    var statuses: [Int]
    var bodies: [String]
    var requests: [URLRequest] = []

    init(statuses: [Int] = [200], bodies: [String] = ["data: {\"type\":\"done\"}\n\n"]) {
        self.statuses = statuses
        self.bodies = bodies
    }

    func connect(_ request: URLRequest) async throws -> SSEConnection {
        requests.append(request)
        let index = min(requests.count - 1, statuses.count - 1)
        return SSEConnection(status: statuses[index], chunks: stream(bodies[min(index, bodies.count - 1)]))
    }

    func recorded() -> [URLRequest] { requests }
}

@Test func parsesEveryUTF8SplitAndLineEnding() throws {
    let wire = "\u{FEFF}: ping\r\ndata: {\"type\":\"speech\",\"text\":\"café 🌳\"}\r\rdata:{\"type\":\"done\"}\n\n"
    let data = Data(wire.utf8)
    for split in 0...data.count {
        var parser = SSEParser()
        let first = try parser.feed(data.prefix(split))
        let second = try parser.feed(data.suffix(data.count - split))
        #expect(first + second == [.speech("café 🌳"), .done])
    }
}

@Test func supportsMultilineDataAndIgnoresMetadata() throws {
    var parser = SSEParser()
    let events = try parser.feed(Data("event: ignored\nid: 1\nretry: 10\ndata: {\ndata: \"type\":\"done\"}\n\n".utf8))
    #expect(events == [.done])
    #expect(try parser.feed(Data("data: {\"type\":\"done\"}".utf8)).isEmpty)
}

@Test func rejectsInvalidAndOversizedData() throws {
    var invalid = SSEParser()
    #expect(throws: CloudError.invalidUTF8) { try invalid.feed(Data([0xff, 10])) }
    var tiny = SSEParser(maximumEventBytes: 4)
    #expect(throws: CloudError.eventTooLarge) { try tiny.feed(Data("12345".utf8)) }
    var aggregate = SSEParser(maximumEventBytes: 20)
    #expect(throws: CloudError.eventTooLarge) {
        try aggregate.feed(Data("data: 1234567890\ndata: 1234567890\n".utf8))
    }
    var json = SSEParser()
    #expect(throws: (any Error).self) { try json.feed(Data("data: {}\n\n".utf8)) }
}

@Test func postsContractAndTokenAndFinishesOnDone() async throws {
    let transport = Transport()
    let client = CloudAnswerer(baseURL: URL(string: "https://example.test/api")!, writeToken: "test-write", transport: transport)
    let request = AskRequest(id: "same-question", question: "What is that?")
    var events: [AskEvent] = []
    for try await event in client.ask(request) { events.append(event) }
    #expect(events == [.status("Connecting…"), .done])
    let sent = await transport.recorded()
    #expect(sent.count == 1)
    #expect(sent[0].url?.path == "/api/v1/ask")
    #expect(sent[0].httpMethod == "POST")
    #expect(sent[0].timeoutInterval == 90)
    #expect(sent[0].value(forHTTPHeaderField: "Authorization") == "Bearer test-write")
    #expect(try JSONDecoder().decode(AskRequest.self, from: sent[0].httpBody!) == request)
}

@Test func retriesTransientFailuresWithIdenticalBody() async throws {
    let transport = Transport(statuses: [503, 200])
    let client = CloudAnswerer(baseURL: URL(string: "https://example.test")!, writeToken: "", transport: transport, retryDelay: .zero)
    var events: [AskEvent] = []
    for try await event in client.ask(AskRequest(id: "retry", question: "Count")) { events.append(event) }
    #expect(events == [.status("Connecting…"), .status("Connecting…"), .done])
    let sent = await transport.recorded()
    #expect(sent.count == 2)
    #expect(sent[0].httpBody == sent[1].httpBody)
}

private actor ColdStartTransport: SSETransport {
    var attempts = 0
    func connect(_ request: URLRequest) async throws -> SSEConnection {
        attempts += 1
        if attempts == 1 { throw URLError(.timedOut) }
        if attempts == 2 { throw URLError(.networkConnectionLost) }
        return SSEConnection(status: 200, chunks: stream("data: {\"type\":\"done\"}\n\n"))
    }
}

@Test func retriesColdStartConnectionFailuresAndReportsConnecting() async throws {
    let transport = ColdStartTransport()
    let client = CloudAnswerer(baseURL: URL(string: "https://fixture.invalid")!, writeToken: "",
        transport: transport, retryDelay: .zero)
    var events: [AskEvent] = []
    for try await event in client.ask(AskRequest(id: "cold", question: "Look")) { events.append(event) }
    #expect(events == [.status("Connecting…"), .status("Connecting…"), .status("Connecting…"), .done])
    #expect(await transport.attempts == 3)
}

@Test func doesNotRetryAuthenticationOrPartialAnswers() async throws {
    for (status, body, expected) in [
        (401, "", CloudError.server(401)),
        (200, "data: {\"type\":\"status\",\"text\":\"Looking\"}\n\n", CloudError.incompleteStream),
    ] {
        let transport = Transport(statuses: [status], bodies: [body])
        let client = CloudAnswerer(baseURL: URL(string: "https://example.test")!, writeToken: "", transport: transport, retryDelay: .zero)
        do {
            for try await _ in client.ask(AskRequest(id: "failure", question: "Look")) {}
            Issue.record("Expected stream error")
        } catch { #expect(error as? CloudError == expected) }
        #expect(await transport.recorded().count == 1)
    }
}

@Test func retriesEmptyDisconnectButBoundsAttempts() async throws {
    let transport = Transport(bodies: [""])
    let client = CloudAnswerer(baseURL: URL(string: "https://example.test")!, writeToken: "", transport: transport, maximumRetries: 2, retryDelay: .zero)
    do {
        for try await _ in client.ask(AskRequest(id: "empty", question: "Look")) {}
        Issue.record("Expected incomplete stream")
    } catch { #expect(error as? CloudError == .incompleteStream) }
    #expect(await transport.recorded().count == 3)
}

private actor HangingTransport: SSETransport {
    var cancelled = false
    var started = false
    func connect(_ request: URLRequest) async throws -> SSEConnection {
        started = true
        do { try await Task.sleep(for: .seconds(60)) }
        catch { cancelled = true; throw error }
        throw CloudError.invalidResponse
    }
}

@Test func consumerCancellationStopsConnection() async throws {
    let transport = HangingTransport()
    let client = CloudAnswerer(baseURL: URL(string: "https://example.test")!, writeToken: "", transport: transport)
    let consumer = Task {
        for try await _ in client.ask(AskRequest(id: "cancel", question: "Look")) {}
    }
    for _ in 0..<100 {
        if await transport.started { break }
        try await Task.sleep(for: .milliseconds(1))
    }
    #expect(await transport.started)
    consumer.cancel()
    _ = await consumer.result
    // The stream's onTermination cancels its producer asynchronously.
    for _ in 0..<100 {
        if await transport.cancelled { return }
        try await Task.sleep(for: .milliseconds(1))
    }
    #expect(await transport.cancelled)
}

private final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func mark() { lock.withLock { value += 1 } }
    func count() -> Int { lock.withLock { value } }
}

private struct ClosingTransport: SSETransport {
    let flag: CancellationFlag
    let status: Int
    func connect(_ request: URLRequest) async throws -> SSEConnection {
        SSEConnection(status: status, chunks: stream("data: {\"type\":\"done\"}\n\n"), cancel: { flag.mark() })
    }
}

@Test func releasesConnectionOnSuccessAndServerError() async throws {
    for status in [200, 401] {
        let flag = CancellationFlag()
        let client = CloudAnswerer(baseURL: URL(string: "https://example.test")!, writeToken: "", transport: ClosingTransport(flag: flag, status: status))
        do {
            for try await _ in client.ask(AskRequest(id: "close", question: "Look")) {}
            #expect(status == 200)
        } catch { #expect(error as? CloudError == .server(401)) }
        #expect(flag.count() == 1)
    }
}
