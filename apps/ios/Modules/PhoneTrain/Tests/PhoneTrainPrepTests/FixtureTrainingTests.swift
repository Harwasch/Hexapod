#if DEBUG
  import Foundation
  import PhoneTrain
  import Testing

  @Test func fixtureTrainingEmitsThreeStepsPreviewAndReproducibleResult() async throws {
    let output = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: output) }
    let png = Data([137, 80, 78, 71])
    let trainer: any PhoneTrainingSession = FixturePhoneTrainingSession(
      outputDirectory: output, previewPNG: png)
    let progress = await trainer.progress()
    let result = try await trainer.run()
    var steps: [Int] = []
    for await value in progress {
      steps.append(try #require(value.sample).step)
      #expect(value.message == "Simulated training")
    }
    #expect(steps == [1, 2, 3])
    #expect(try await trainer.previewImage() == png)
    #expect(result.simulated)
    #expect(result.ply == output.appendingPathComponent("trained.ply"))
    let first = try Data(contentsOf: result.ply)
    #expect(String(decoding: first, as: UTF8.self).contains("element vertex 1"))
    let again = FixturePhoneTrainingSession(outputDirectory: output, previewPNG: png)
    let repeated = try await again.run()
    #expect(try Data(contentsOf: repeated.ply) == first)
    try await trainer.closeAndCleanup()
    try await again.closeAndCleanup()
  }

  @Test func fixtureTrainingResumesWithoutPollingAndCancellationUnblocksInactivity() async throws {
    let output = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: output) }
    let trainer = FixturePhoneTrainingSession(outputDirectory: output, previewPNG: Data())
    await trainer.pause()
    await trainer.setActive(false)
    let work = Task { try await trainer.run() }
    await trainer.resume()
    await trainer.setActive(true)
    #expect(try await work.value.simulated)
    try await trainer.closeAndCleanup()

    let stopped = FixturePhoneTrainingSession(outputDirectory: output, previewPNG: Data())
    await stopped.setActive(false)
    let cancelled = Task { try await stopped.run() }
    cancelled.cancel()
    do {
      _ = try await cancelled.value
      Issue.record("Cancelled inactive trainer completed")
    } catch is CancellationError {}
    try await stopped.closeAndCleanup()

    let preCancelled = FixturePhoneTrainingSession(outputDirectory: output, previewPNG: Data())
    await preCancelled.setActive(false)
    await preCancelled.cancel()
    await #expect(throws: CancellationError.self) { try await preCancelled.run() }
  }
#endif
