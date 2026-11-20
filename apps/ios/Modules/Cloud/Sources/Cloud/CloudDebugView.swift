#if DEBUG && canImport(SwiftUI)
import EntitySchema
import SwiftUI

/// Standalone harness: embed CloudDebugView() in any debug app or open its preview.
@MainActor
public struct CloudDebugView: View {
    @State private var baseURL = "http://127.0.0.1:8000"
    @State private var token = ""
    @State private var scanId = ""
    @State private var question = "How tall is the tree?"
    @State private var useFake = true
    @State private var output: [String] = []
    @State private var task: Task<Void, Never>?

    public init() {}

    public var body: some View {
        Form {
            Text("Cloud client • Simulated fixture replay")
            Toggle("Use bundled fake", isOn: $useFake)
            if !useFake {
                TextField("API URL", text: $baseURL)
                SecureField("Write token", text: $token)
                TextField("Capture UUID (required by server replay)", text: $scanId)
            }
            TextField("Question", text: $question)
            Button("Ask") { run() }.disabled(task != nil)
            Button("Cancel") { task?.cancel() }
            ForEach(Array(output.enumerated()), id: \.offset) { _, text in
                Text(text)
            }
        }
        .onDisappear { task?.cancel() }
    }

    private func run() {
        output = []
        task = Task {
            defer { task = nil }
            do {
                let answerer: any DeepAnswering
                if useFake {
                    answerer = try FakeDeepAnswering.yardSmall()
                } else {
                    guard let url = URL(string: baseURL),
                          ["http", "https"].contains(url.scheme ?? ""), url.host != nil
                    else { throw CloudError.invalidResponse }
                    answerer = CloudAnswerer(baseURL: url, writeToken: token)
                }
                let request = AskRequest(
                    id: UUID().uuidString, question: question,
                    scanId: scanId.isEmpty ? nil : scanId
                )
                for try await event in answerer.ask(request) {
                    switch event {
                    case .status(let text), .speech(let text): output.append(text)
                    case .entity(let entity):
                        output.append("\(entity.label): observed height \(entity.extent.y) m")
                    case .done: output.append("Done")
                    }
                }
            } catch { output.append(String(describing: error)) }
        }
    }
}

struct CloudDebugView_Previews: PreviewProvider {
    static var previews: some View { CloudDebugView() }
}
#endif
