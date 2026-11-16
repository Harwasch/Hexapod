import Testing
@testable import Router

@Test func placeholderCanBeConstructed() {
    let value = RouterPlaceholder()
    #expect(type(of: value) == RouterPlaceholder.self)
}
