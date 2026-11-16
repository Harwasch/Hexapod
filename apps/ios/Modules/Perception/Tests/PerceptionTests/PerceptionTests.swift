import Testing
@testable import Perception

@Test func placeholderCanBeConstructed() {
    let value = PerceptionPlaceholder()
    #expect(type(of: value) == PerceptionPlaceholder.self)
}
