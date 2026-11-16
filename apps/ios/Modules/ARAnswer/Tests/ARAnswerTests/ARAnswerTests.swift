import Testing
@testable import ARAnswer

@Test func placeholderCanBeConstructed() {
    let value = ARAnswerPlaceholder()
    #expect(type(of: value) == ARAnswerPlaceholder.self)
}
