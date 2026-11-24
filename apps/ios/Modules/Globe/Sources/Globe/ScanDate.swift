import Foundation

/// A scan menu shows a local, short date; malformed dates never leak a raw server timestamp.
public func globeScanDate(_ timestamp: String, locale: Locale = .current) -> String {
  let parser = ISO8601DateFormatter()
  parser.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
  let date = parser.date(from: timestamp) ?? ISO8601DateFormatter().date(from: timestamp)
  guard let date else { return "Scan" }
  let formatter = DateFormatter()
  formatter.locale = locale
  formatter.dateStyle = .medium
  formatter.timeStyle = .none
  return formatter.string(from: date)
}
