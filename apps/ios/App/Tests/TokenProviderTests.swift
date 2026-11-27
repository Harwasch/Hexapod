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

#if DEBUG
  struct UITestTokenStoreTests {
    @Test func ordinaryLaunchKeepsKeychainStorage() {
      #expect(APICredentials.makeUITestTokenStore(mode: LaunchMode(arguments: [])) == nil)
    }

    @Test func fixtureCredentialsAreTrimmedSharedAndRemovable() throws {
      for fixture in ["yard-small", "production-providers"] {
        let mode = LaunchMode(arguments: ["Hex", "-uiTestFixture", fixture])
        let store = try #require(APICredentials.makeUITestTokenStore(mode: mode))
        #expect(store.writeToken.isEmpty)
        store.setWriteToken("  fixture-token\n")
        let backgroundReader = store
        #expect(backgroundReader.writeToken == "fixture-token")
        store.setWriteToken(" \n")
        #expect(backgroundReader.writeToken.isEmpty)
      }
    }

    @Test func fixtureCredentialsDoNotSurviveAnotherLaunch() throws {
      let mode = LaunchMode(arguments: ["Hex", "-uiTestFixture", "yard-small"])
      let store = try #require(APICredentials.makeUITestTokenStore(mode: mode))
      store.setWriteToken("fixture-token")
      let relaunched = try #require(APICredentials.makeUITestTokenStore(mode: mode))
      #expect(relaunched.writeToken.isEmpty)
      #expect(store.writeToken == "fixture-token")
    }
  }
#endif
