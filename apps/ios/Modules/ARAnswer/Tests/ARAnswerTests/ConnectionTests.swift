import EntitySchema
import Testing

@testable import ARAnswer

@Test func connectionStatusClearsOnReplyCompletionAndReset() {
  var value = AnswerPresentation()
  value.receive(.status("Connecting…"))
  #expect(value.isConnecting)
  #expect(!value.isUnavailable)
  value.receive(.speech("Answer"))
  #expect(!value.isConnecting)
  value.receive(.status("Connecting…"))
  value.receive(.done)
  #expect(!value.isConnecting)
  value.receive(.status("Connecting…"))
  value.reset()
  #expect(!value.isConnecting)
}
