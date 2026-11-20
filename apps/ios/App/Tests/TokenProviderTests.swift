import Foundation
import Testing

@testable import AppSupport

struct TokenProviderTests {
  @Test func pastedCredentialWins() {
    #expect(TokenProvider.resolve(pasted: "  pasted-key\n", bundled: "phone-key") == "pasted-key")
    #expect(TokenProvider.resolve(pasted: "write-key", bundled: nil) == "write-key")
  }

  @Test func fallsBackToBuildCredential() {
    #expect(TokenProvider.resolve(pasted: "", bundled: "phone-key") == "phone-key")
    #expect(TokenProvider.resolve(pasted: " \n", bundled: " phone-key ") == "phone-key")
  }

  @Test func emptyOrUnexpandedBuildCredentialIsUnavailable() {
    for value: String? in [nil, "", " \n", "$(HEX_API_PHONE_TOKEN)", " $(HEX_API_PHONE_TOKEN) "] {
      #expect(TokenProvider.resolve(pasted: "", bundled: value) == nil)
    }
  }
}
