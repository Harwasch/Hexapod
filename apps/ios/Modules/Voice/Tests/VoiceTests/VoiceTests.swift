import Testing
@testable import Voice

@Test func placeholderCanBeConstructed() {
    let value = VoicePlaceholder()
    #expect(type(of: value) == VoicePlaceholder.self)
}
