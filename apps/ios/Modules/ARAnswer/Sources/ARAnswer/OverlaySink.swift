import EntitySchema
import Foundation

@MainActor public protocol AnswerOverlaySink: AnyObject {
  func update(entities: [Entity])
  func setMask(_ mask: HighlightMask?)
  func speak(_ text: String)
  func stopSpeaking()
  func remove()
}

@MainActor public final class FakeOverlaySink: AnswerOverlaySink {
  public private(set) var draws: [[Entity]] = []
  public private(set) var masks: [HighlightMask?] = []
  public private(set) var utterances: [String] = []
  public private(set) var stopCount = 0
  public private(set) var removeCount = 0
  public init() {}
  public func update(entities: [Entity]) { draws.append(entities) }
  public func setMask(_ mask: HighlightMask?) { masks.append(mask) }
  public func speak(_ text: String) { utterances.append(text) }
  public func stopSpeaking() { stopCount += 1 }
  public func remove() { removeCount += 1 }
}

#if canImport(SwiftUI)
  import SwiftUI
  public struct ARAnswerDebugScreen: View {
    @State private var result = "Simulated overlay sink"
    public init() {}
    public var body: some View {
      VStack {
        Text(result)
        Button("Record drawing and speech") {
          let sink = FakeOverlaySink()
          sink.update(entities: [])
          sink.setMask(.init(width: 2, height: 2, pixels: [255, 0, 0, 255]))
          sink.speak("Simulated answer")
          result =
            "Simulated: \(sink.draws.count) draw, \(sink.masks.count) mask, \(sink.utterances)"
        }
      }.padding()
    }
  }
  #Preview { ARAnswerDebugScreen() }
#endif
