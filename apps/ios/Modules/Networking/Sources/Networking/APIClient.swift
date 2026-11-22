import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// One foreground API session for catalog reads, upload control and streamed answers.
public enum APISession {
  public static func configuration() -> URLSessionConfiguration {
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = 90
    // Also bounds waitsForConnectivity, which otherwise can wait for days.
    config.timeoutIntervalForResource = 180
    #if canImport(Darwin)
      config.waitsForConnectivity = true
    #endif
    return config
  }
  public static let shared = URLSession(configuration: configuration())
}

public enum APIError: Error, Equatable {
  case invalidResponse
  case httpStatus(Int)
}

public enum RetryPolicy {
  public static let maximumRetries = 2
  public static func retryable(status: Int) -> Bool {
    status == 408 || status == 429 || (500...599).contains(status)
  }
  public static func retryable(_ error: any Error) -> Bool {
    if case APIError.httpStatus(let status) = error { return retryable(status: status) }
    guard let error = error as? URLError else { return false }
    return [
      .timedOut, .cannotFindHost, .cannotConnectToHost, .networkConnectionLost,
      .dnsLookupFailed, .notConnectedToInternet,
    ].contains(error.code)
  }
  public static func delay(attempt: Int) -> Duration {
    .seconds(1 << min(4, max(0, attempt)))
  }
}

public protocol APITransport: Sendable {
  func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse)
}

public struct URLSessionAPITransport: APITransport {
  public init() {}
  public func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
    let (data, response) = try await APISession.shared.data(for: request)
    guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
    return (data, http)
  }
}

/// Only use retries for reads or endpoints with an explicit idempotency key.
public struct APIClient: Sendable {
  private let transport: any APITransport
  private let sleep: @Sendable (Duration) async throws -> Void
  public init(
    transport: any APITransport = URLSessionAPITransport(),
    sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
  ) {
    self.transport = transport
    self.sleep = sleep
  }
  public func data(for request: URLRequest) async throws -> Data {
    var request = request
    request.timeoutInterval = 90
    for attempt in 0...RetryPolicy.maximumRetries {
      try Task.checkCancellation()
      do {
        let (data, response) = try await transport.data(for: request)
        guard (200..<300).contains(response.statusCode) else {
          throw APIError.httpStatus(response.statusCode)
        }
        return data
      } catch {
        try Task.checkCancellation()
        guard attempt < RetryPolicy.maximumRetries, RetryPolicy.retryable(error) else {
          throw error
        }
        try await sleep(RetryPolicy.delay(attempt: attempt))
      }
    }
    throw APIError.invalidResponse
  }
}
