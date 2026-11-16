import Foundation
import Testing
@testable import EntitySchema

private func fixture() throws -> Data {
    let url = try #require(Bundle.module.url(forResource: "entity", withExtension: "json", subdirectory: "Fixtures"))
    return try Data(contentsOf: url)
}

@Test func fixtureRoundTripsAllJSONKinds() throws {
    let entity = try JSONDecoder().decode(Entity.self, from: fixture())
    #expect(entity.label == "oak")
    #expect(entity.anchor.world.z == -2.5)
    #expect(entity.anchor.geo?.lat == 34.6)
    #expect(entity.masks == [MaskReference(keyframe: "000001", ref: "masks/000001.png")])
    #expect(entity.attrs["count"] == .integer(9007199254740993))
    #expect(entity.attrs["reviewed"] == .bool(true))
    #expect(entity.attrs["note"] == .null)
    #expect(entity.attrs["tags"] == .array([.string("tree"), .number(3.5), .object(["season": .string("autumn")])]))
    let encoded = try JSONEncoder().encode(entity)
    #expect(try JSONDecoder().decode(Entity.self, from: encoded) == entity)
}

@Test func localAnchorOmitsGeo() throws {
    var entity = try JSONDecoder().decode(Entity.self, from: fixture())
    entity.anchor.geo = nil
    entity.masks = []
    entity.attrs = [:]
    let data = try JSONEncoder().encode(entity)
    #expect(try JSONDecoder().decode(Entity.self, from: data) == entity)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let anchor = try #require(json["anchor"] as? [String: Any])
    #expect(anchor["geo"] == nil)
}

@Test func missingRequiredFieldFailsDecoding() throws {
    var json = try #require(JSONSerialization.jsonObject(with: fixture()) as? [String: Any])
    json.removeValue(forKey: "source")
    let data = try JSONSerialization.data(withJSONObject: json)
    #expect(throws: DecodingError.self) { try JSONDecoder().decode(Entity.self, from: data) }
}

@Test func numericRoundTripPreservesJSONMeaningWithoutRoundingIDs() throws {
    let values: [JSONValue] = [.number(3.0), .number(-0.0), .integer(Int64.max), .number(3.5)]
    for value in values {
        let encoded = try JSONEncoder().encode(value)
        #expect(try JSONDecoder().decode(JSONValue.self, from: encoded) == value)
    }
    #expect(JSONValue.integer(9007199254740993) != .number(9007199254740992.0))
    #expect(JSONValue.bool(true) != .integer(1))
}
