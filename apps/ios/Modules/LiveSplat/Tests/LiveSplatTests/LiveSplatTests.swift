import Testing
@testable import LiveSplat

@Test func placeholderCanBeConstructed() {
    let value = LiveSplatPlaceholder()
    #expect(type(of: value) == LiveSplatPlaceholder.self)
}
