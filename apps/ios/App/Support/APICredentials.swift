import Foundation

#if canImport(Security)
  import Security
#endif

/// Resolves the effective API credential; Settings edits only the pasted Keychain value.
struct TokenProvider {
  static func resolve(pasted: String, bundled: String?) -> String? {
    let pasted = pasted.trimmingCharacters(in: .whitespacesAndNewlines)
    if !pasted.isEmpty { return pasted }
    let bundled = (bundled ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    return bundled.isEmpty || bundled.contains("$(") ? nil : bundled
  }
}

/// Shared by Settings and background uploads. Pasted tokens stay in the Keychain.
/// DEBUG UI-test launches use a process-local store because CI simulators are unsigned.
enum APICredentials {
  #if DEBUG
    private static let uiTestTokenStore = makeUITestTokenStore(mode: LaunchMode.current)

    static func makeUITestTokenStore(mode: LaunchMode) -> UITestTokenStore? {
      mode.isUITest ? UITestTokenStore() : nil
    }
  #endif

  static var token: String {
    TokenProvider.resolve(
      pasted: writeToken,
      bundled: Bundle.main.object(forInfoDictionaryKey: "HEX_API_PHONE_TOKEN") as? String
    ) ?? ""
  }
  enum CredentialError: Error { case unavailable, keychain(Int32) }
  static var writeToken: String {
    #if DEBUG
      if let uiTestTokenStore { return uiTestTokenStore.writeToken }
    #endif
    #if canImport(Security)
      var query = item
      query[kSecReturnData as String] = true
      query[kSecMatchLimit as String] = kSecMatchLimitOne
      var result: CFTypeRef?
      guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
        let data = result as? Data
      else { return "" }
      return String(decoding: data, as: UTF8.self)
    #else
      return ""
    #endif
  }
  static func setWriteToken(_ value: String) throws {
    #if DEBUG
      if let uiTestTokenStore {
        uiTestTokenStore.setWriteToken(value)
        return
      }
    #endif
    #if canImport(Security)
      let token = value.trimmingCharacters(in: .whitespacesAndNewlines)
      if token.isEmpty {
        let status = SecItemDelete(item as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
          throw CredentialError.keychain(status)
        }
        return
      }
      let attributes: [String: Any] = [
        kSecValueData as String: Data(token.utf8),
        kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
      ]
      let status = SecItemUpdate(item as CFDictionary, attributes as CFDictionary)
      if status == errSecItemNotFound {
        let added = SecItemAdd(item.merging(attributes) { _, new in new } as CFDictionary, nil)
        guard added == errSecSuccess else { throw CredentialError.keychain(added) }
      } else if status != errSecSuccess {
        throw CredentialError.keychain(status)
      }
    #else
      throw CredentialError.unavailable
    #endif
  }
  #if canImport(Security)
    private static var item: [String: Any] {
      [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: "hex.api", kSecAttrAccount as String: "write-token",
      ]
    }
  #endif
}

#if DEBUG
  /// One instance per UI-test process, shared by Settings and background readers.
  /// Every access is protected by the lock; nothing is written to disk or Keychain.
  final class UITestTokenStore: Sendable {
    private let lock = NSLock()
    private nonisolated(unsafe) var token = ""

    var writeToken: String { lock.withLock { token } }

    func setWriteToken(_ value: String) {
      lock.withLock { token = value.trimmingCharacters(in: .whitespacesAndNewlines) }
    }
  }
#endif
