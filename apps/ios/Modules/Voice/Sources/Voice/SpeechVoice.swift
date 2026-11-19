#if canImport(Speech) && canImport(AVFoundation) && os(iOS)
  @preconcurrency import AVFoundation
  @preconcurrency import Speech
  import Foundation

  /// Unverified on Linux: SpeechAnalyzer SDK signatures, audio conversion, asset installation,
  /// interruption recovery and final-result latency require Xcode 26 and a physical iPhone.
  @available(iOS 26.0, *)
  @MainActor public final class SpeechVoice: VoiceInput {
    private let locale: Locale
    private var engine: AVAudioEngine?
    private var analyzer: SpeechAnalyzer?
    private var input: AsyncStream<AnalyzerInput>.Continuation?
    private var output: AsyncThrowingStream<Transcript, Error>.Continuation?
    private var recognition: Task<Void, Never>?
    private var analysis: Task<Void, Never>?
    private var interruption: NSObjectProtocol?
    private var generation = UUID()

    public init(locale: Locale = .current) { self.locale = locale }

    public func start() async throws -> AsyncThrowingStream<Transcript, Error> {
      cancel()
      let id = generation
      guard await AVAudioApplication.requestRecordPermission() else {
        throw VoiceError.permissionDenied
      }
      guard id == generation, !Task.isCancelled else { throw CancellationError() }
      let transcriber = SpeechTranscriber(
        locale: locale, transcriptionOptions: [],
        reportingOptions: [.volatileResults], attributeOptions: [])
      guard SpeechTranscriber.isAvailable else { throw VoiceError.unavailable }
      if let installation = try await AssetInventory.assetInstallationRequest(supporting: [
        transcriber
      ]) {
        try await installation.downloadAndInstall()
      }
      guard id == generation, !Task.isCancelled else { throw CancellationError() }
      guard
        let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
      else {
        throw VoiceError.unavailable
      }
      guard id == generation, !Task.isCancelled else { throw CancellationError() }
      let session = AVAudioSession.sharedInstance()
      try session.setCategory(
        .playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .duckOthers])
      try session.setActive(true)
      do {
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let engine = AVAudioEngine()
        let audio = AsyncStream<AnalyzerInput>.makeStream(bufferingPolicy: .bufferingNewest(64))
        let results = AsyncThrowingStream<Transcript, Error>.makeStream()
        let bridge = try AudioBridge(
          source: engine.inputNode.outputFormat(forBus: 0), target: format)
        self.engine = engine
        self.analyzer = analyzer
        input = audio.continuation
        output = results.continuation
        results.continuation.onTermination = { [weak self] _ in
          Task { @MainActor in if self?.generation == id { self?.cancel() } }
        }
        recognition = Task { [weak self] in
          do {
            for try await result in transcriber.results {
              try Task.checkCancellation()
              guard self?.generation == id else { return }
              results.continuation.yield(
                Transcript(text: String(result.text.characters), isFinal: result.isFinal))
            }
            results.continuation.finish()
          } catch { results.continuation.finish(throwing: error) }
        }
        analysis = Task { [weak self] in
          do { try await analyzer.start(inputSequence: audio.stream) } catch {
            guard self?.generation == id else { return }
            results.continuation.finish(throwing: error)
          }
        }
        // The audio thread invokes this block; it must not inherit MainActor isolation.
        engine.inputNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { @Sendable buffer, _ in
          do {
            let converted = try bridge.convert(buffer)
            if case .dropped = audio.continuation.yield(AnalyzerInput(buffer: converted)) {
              // Dropped audio invalidates the utterance; never silently answer a truncated question.
              results.continuation.finish(throwing: VoiceError.interrupted)
              audio.continuation.finish()
            }
          } catch {
            results.continuation.finish(throwing: error)
            audio.continuation.finish()
          }
        }
        interruption = NotificationCenter.default.addObserver(
          forName: AVAudioSession.interruptionNotification, object: session, queue: .main
        ) { [weak self] _ in
          Task { @MainActor in
            guard self?.generation == id else { return }
            self?.output?.finish(throwing: VoiceError.interrupted)
            self?.cancel()
          }
        }
        engine.prepare()
        try engine.start()
        return results.stream
      } catch {
        cancel()
        throw error
      }
    }

    public func stop() async {
      let id = generation
      guard let analyzer else {
        cancel()
        return
      }
      let pendingRecognition = recognition
      stopAudio()
      input?.finish()
      input = nil
      do { try await analyzer.finalizeAndFinishThroughEndOfInput() } catch {
        if id == generation { output?.finish(throwing: error) }
      }
      await pendingRecognition?.value
      guard id == generation else { return }
      cancel()
    }

    public func cancel() {
      generation = UUID()
      stopAudio()
      input?.finish()
      input = nil
      output?.finish()
      output = nil
      recognition?.cancel()
      recognition = nil
      analysis?.cancel()
      analysis = nil
      if let analyzer { Task { await analyzer.cancelAndFinishNow() } }
      analyzer = nil
    }

    private func stopAudio() {
      if let engine {
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
      }
      engine = nil
      if let interruption { NotificationCenter.default.removeObserver(interruption) }
      interruption = nil
      try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
  }

  /// AVAudioEngine invokes its tap serially. Converter and its buffers never leave that
  /// callback; only the newly allocated, converted buffer is transferred to AnalyzerInput.
  private final class AudioBridge: @unchecked Sendable {
    private let converter: AVAudioConverter
    private let target: AVAudioFormat
    init(source: AVAudioFormat, target: AVAudioFormat) throws {
      guard let converter = AVAudioConverter(from: source, to: target) else {
        throw VoiceError.unavailable
      }
      self.converter = converter
      self.target = target
    }
    func convert(_ source: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
      let capacity = AVAudioFrameCount(
        ceil(Double(source.frameLength) * target.sampleRate / source.format.sampleRate) + 32)
      guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
        throw VoiceError.unavailable
      }
      var sent = false
      var error: NSError?
      let status = converter.convert(to: output, error: &error) { _, state in
        if sent {
          state.pointee = .noDataNow
          return nil
        }
        sent = true
        state.pointee = .haveData
        return source
      }
      if let error { throw error }
      guard status != .error else { throw VoiceError.unavailable }
      return output
    }
  }
#endif
