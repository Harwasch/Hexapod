import Foundation
import Testing
@testable import EntitySchema

private func entity() throws -> Entity {
    let url = try #require(Bundle.module.url(forResource: "entity", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode(Entity.self, from: Data(contentsOf: url))
}

private func json(_ value: some Encodable) throws -> [String: Any] {
    let data = try JSONEncoder().encode(value)
    return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

@Test func requestRoundTripsAndOmitsAbsentOptionals() throws {
    let request = AskRequest(
        id: "q-1", question: "how tall is that oak",
        keyframes: [KeyframeRef(index: 42, image: "frames/000042.png", pose: "frames.jsonl#42")],
        deviceEntities: [try entity()])
    let data = try JSONEncoder().encode(request)
    #expect(try JSONDecoder().decode(AskRequest.self, from: data) == request)
    let object = try json(request)
    #expect(object["scanId"] == nil)
    #expect(object["location"] == nil)
    let frame = try #require((object["keyframes"] as? [[String: Any]])?.first)
    #expect(frame["depth"] == nil)
}

@Test func eventsUseATypeTag() throws {
    #expect(try json(AskEvent.speech("About 14 metres."))["type"] as? String == "speech")
    #expect(try json(AskEvent.status("measuring"))["text"] as? String == "measuring")
    #expect(try json(AskEvent.done).keys.sorted() == ["type"])
    let tagged = try json(AskEvent.entity(try entity()))
    #expect(tagged["type"] as? String == "entity")
    #expect(tagged["entity"] is [String: Any])
    for event in [AskEvent.entity(try entity()), .speech("a"), .status("b"), .done] {
        let data = try JSONEncoder().encode(event)
        #expect(try JSONDecoder().decode(AskEvent.self, from: data) == event)
    }
}

@Test func unknownEventTypeFails() throws {
    let data = Data(#"{"type":"shout","text":"x"}"#.utf8)
    #expect(throws: DecodingError.self) { try JSONDecoder().decode(AskEvent.self, from: data) }
}

private struct Canned: DeepAnswering {
    func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.status("thinking"))
            continuation.yield(.speech(request.question))
            continuation.yield(.done)
            continuation.finish()
        }
    }
}

@Test func deepAnsweringStreamsInOrder() async throws {
    var seen: [AskEvent] = []
    for try await event in Canned().ask(AskRequest(id: "1", question: "hi")) { seen.append(event) }
    #expect(seen == [.status("thinking"), .speech("hi"), .done])
}
