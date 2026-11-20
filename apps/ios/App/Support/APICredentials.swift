import Foundation

#if canImport(Security)
  import Security
#endif

/// Shared by Settings and background uploads. Tokens never go in preferences or the bundle.
enum APICredentials {
  enum CredentialError: Error { case unavailable, keychain(Int32) }
  static var writeToken: String {
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
