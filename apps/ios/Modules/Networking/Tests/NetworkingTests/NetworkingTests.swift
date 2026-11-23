import Foundation
import Testing

@testable import Networking

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

private actor FakeTransport: APITransport {
  var attempts = 0
  let outcomes: [Int]
  init(_ outcomes: [Int]) { self.outcomes = outcomes }
  func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
    let status = outcomes[min(attempts, outcomes.count - 1)]
    attempts += 1
    if status == 0 { throw URLError(.timedOut) }
    if status == -1 { throw URLError(.networkConnectionLost) }
    return (
      Data("ok".utf8),
      HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
    )
  }
}

private actor Delays {
  var values: [Duration] = []
  func append(_ delay: Duration) { values.append(delay) }
}

@Test func backoffUsesOneThenTwoSecondsAndCancellationStopsBeforeAnotherRequest() async throws {
  let delays = Delays()
  let transport = FakeTransport([503, 502, 200])
  let client = APIClient(transport: transport, sleep: { await delays.append($0) })
  _ = try await client.data(for: URLRequest(url: URL(string: "https://fixture.invalid")!))
  #expect(await delays.values == [.seconds(1), .seconds(2)])
  let failing = FakeTransport([503, 200])
  let cancelled = APIClient(transport: failing, sleep: { _ in throw CancellationError() })
  await #expect(throws: CancellationError.self) {
    _ = try await cancelled.data(for: URLRequest(url: URL(string: "https://fixture.invalid")!))
  }
  #expect(await failing.attempts == 1)
}

@Test func coldStartRetriesTimeoutServerAndConnectionErrorsButStops() async throws {
  for outcomes in [[0, 503, 200], [-1, 200], [500, 500, 500]] {
    let transport = FakeTransport(outcomes)
    let client = APIClient(transport: transport, sleep: { _ in })
    do {
      _ = try await client.data(for: URLRequest(url: URL(string: "https://fixture.invalid")!))
      #expect(outcomes.last == 200)
    } catch { #expect(outcomes.last == 500) }
    #expect(await transport.attempts == outcomes.count)
  }
}

@Test func refusesAuthenticationRetryAndHonorsCancellation() async {
  let transport = FakeTransport([401])
  let client = APIClient(transport: transport, sleep: { _ in })
  await #expect(throws: APIError.httpStatus(401)) {
    _ = try await client.data(for: URLRequest(url: URL(string: "https://fixture.invalid")!))
  }
  #expect(await transport.attempts == 1)
  let task = Task {
    try Task.checkCancellation()
    _ = try await client.data(for: URLRequest(url: URL(string: "https://fixture.invalid")!))
  }
  task.cancel()
  _ = await task.result
  #expect(!RetryPolicy.retryable(URLError(.cancelled)))
  #expect(!RetryPolicy.retryable(URLError(.serverCertificateUntrusted)))
}

@Test func sessionAndBackoffAreBounded() {
  let config = APISession.configuration()
  #expect(config.timeoutIntervalForRequest == 90)
  #expect(config.timeoutIntervalForResource == 180)
  #if canImport(Darwin)
    #expect(config.waitsForConnectivity)
  #endif
  #expect(RetryPolicy.delay(attempt: 0) == .seconds(1))
  #expect(RetryPolicy.delay(attempt: 1) == .seconds(2))
}
