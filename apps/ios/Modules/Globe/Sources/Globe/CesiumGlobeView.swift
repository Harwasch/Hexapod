#if canImport(SwiftUI) && canImport(WebKit) && canImport(UIKit)
  import SwiftUI
  import UIKit
  import WebKit

  /// Both renderers consume the same native snapshot and return native IDs.
  @MainActor
  public protocol GlobeRenderer: View {
    var snapshot: GlobeSnapshot { get }
  }

  /// One retained HTTPS page. Explicit fixtures use the bundled copy.
  @MainActor
  public final class CesiumGlobeSession: NSObject, ObservableObject, WKScriptMessageHandler,
    WKNavigationDelegate
  {
    public static let shared = CesiumGlobeSession()
    public let webView: WKWebView
    @Published public private(set) var error: String?
    private var origin: GlobeOrigin?
    private var bridge = GlobeBridgeState()
    private var snapshot: GlobeSnapshot?
    private var owner: UUID?
    private var onPin: ((GlobePin) -> Void)?
    private var onScan: ((UUID) -> Void)?

    public override init() {
      let configuration = WKWebViewConfiguration()
      configuration.websiteDataStore = .nonPersistent()
      webView = WKWebView(frame: .zero, configuration: configuration)
      super.init()
      configuration.userContentController.add(WeakGlobeHandler(self), name: "globe")
      webView.navigationDelegate = self
      webView.isOpaque = false
      webView.scrollView.isScrollEnabled = false
    }
    public func configure(webBaseURL: URL?, fixture: Bool) {
      let local = fixture ? Bundle.module.url(
        forResource: "globe-embed", withExtension: "html", subdirectory: "GlobeWeb") : nil
      guard !fixture || local != nil else { error = "Globe fixture unavailable"; return }
      let next = GlobeOrigin(webBaseURL: webBaseURL ?? GlobeOrigin.deployed, fixturePage: local)
      guard origin?.page != next.page else { return }
      origin = next
      load()
    }
    public func focus(scanID: UUID) {
      for message in bridge.openScan(scanID.uuidString.lowercased()) { send(message) }
    }
    public func load() {
      bridge.invalidate()
      error = nil
      guard let page = origin?.page else { error = "Globe unavailable"; return }
      if page.isFileURL {
        webView.loadFileURL(page, allowingReadAccessTo: page.deletingLastPathComponent())
      } else {
        webView.load(URLRequest(url: page))
      }
    }
    func update(
      owner: UUID, snapshot: GlobeSnapshot, scanID: UUID?, fixture: Bool, webBaseURL: URL? = nil,
      onPin: @escaping (GlobePin) -> Void, onScan: @escaping (UUID) -> Void
    ) {
      configure(webBaseURL: webBaseURL, fixture: fixture)
      self.owner = owner
      self.snapshot = snapshot
      self.onPin = onPin
      self.onScan = onScan
      for message in bridge.update(
        snapshot: snapshot.bridgeSnapshot(fixture: fixture),
        scanID: scanID?.uuidString.lowercased())
      { send(message) }
    }
    func detach(owner: UUID) {
      guard self.owner == owner else { return }
      onPin = nil
      onScan = nil
      self.owner = nil
    }
    private func send(_ message: BridgeMessage) {
      guard bridge.ready, let data = try? JSONEncoder().encode(message),
        let json = String(data: data, encoding: .utf8)
      else { return }
      // JSON is passed as an argument, never interpolated into executable JavaScript.
      webView.callAsyncJavaScript(
        "window.globeBridge.receive(JSON.parse(payload))",
        arguments: ["payload": json], in: nil, in: .page
      ) { [weak self] result in
        if case .failure = result { self?.error = "Globe communication failed" }
      }
    }
    public func userContentController(
      _ userContentController: WKUserContentController, didReceive message: WKScriptMessage
    ) {
      let sender = message.frameInfo.securityOrigin
      guard message.webView === webView, message.frameInfo.isMainFrame,
        origin?.allowsMessage(message.frameInfo.request.url) == true,
        origin?.allowsSecurityOrigin(scheme: sender.`protocol`, host: sender.host, port: sender.port) == true
      else {
        return
      }
      do {
        let data = try JSONSerialization.data(withJSONObject: message.body)
        let event = try JSONDecoder().decode(BridgeMessage.self, from: data)
        let action = try GlobeBridgeDispatcher().dispatch(event, snapshot: snapshot)
        switch action {
        case .ready:
          for message in bridge.becomeReady() { send(message) }
        case .pin(let id): if let pin = snapshot?.pins.first(where: { $0.id == id }) { onPin?(pin) }
        case .scan(let id):
          error = nil
          onScan?(id)
        case .failure(let message): error = message
        }
      } catch { self.error = "Invalid globe event" }
    }
    public func webViewWebContentProcessDidTerminate(_ webView: WKWebView) { load() }
    public func webView(
      _ webView: WKWebView, didFail navigation: WKNavigation!, withError error: any Error
    ) {
      self.error = "Globe could not load"
    }
    public func webView(
      _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
      withError error: any Error
    ) {
      self.error = "Globe could not load"
    }
    public func webView(
      _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
      decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
    ) {
      decisionHandler(
        navigationAction.targetFrame?.isMainFrame == true
          && origin?.allowsNavigation(navigationAction.request.url) == true ? .allow : .cancel)
    }
  }

  @MainActor
  private final class WeakGlobeHandler: NSObject, WKScriptMessageHandler {
    weak var owner: CesiumGlobeSession?
    init(_ owner: CesiumGlobeSession) { self.owner = owner }
    func userContentController(
      _ controller: WKUserContentController, didReceive message: WKScriptMessage
    ) {
      owner?.userContentController(controller, didReceive: message)
    }
  }

  @MainActor
  public struct CesiumGlobeView: GlobeRenderer {
    public let snapshot: GlobeSnapshot
    let scanID: UUID?
    let fixture: Bool
    let webBaseURL: URL?
    let onPin: (GlobePin) -> Void
    let onScan: (UUID) -> Void
    @ObservedObject private var session = CesiumGlobeSession.shared
    public init(
      snapshot: GlobeSnapshot, scanID: UUID? = nil, fixture: Bool = false, webBaseURL: URL? = nil,
      onPin: @escaping (GlobePin) -> Void, onScan: @escaping (UUID) -> Void
    ) {
      self.snapshot = snapshot
      self.scanID = scanID
      self.fixture = fixture
      self.webBaseURL = webBaseURL
      self.onPin = onPin
      self.onScan = onScan
    }
    public var body: some View {
      ZStack(alignment: .top) {
        GlobeWebSurface(
          session: session, snapshot: snapshot, scanID: scanID, fixture: fixture, webBaseURL: webBaseURL, onPin: onPin,
          onScan: onScan)
        if let error = session.error {
          VStack {
            Text(error)
            Button("Retry") { session.load() }
          }.padding().background(.regularMaterial)
        }
      }
    }
  }

  private struct GlobeWebSurface: UIViewRepresentable {
    let session: CesiumGlobeSession
    let snapshot: GlobeSnapshot
    let scanID: UUID?
    let fixture: Bool
    let webBaseURL: URL?
    let onPin: (GlobePin) -> Void
    let onScan: (UUID) -> Void
    final class Coordinator { let id = UUID() }
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeUIView(context: Context) -> GlobeContainer { GlobeContainer() }
    func updateUIView(_ view: GlobeContainer, context: Context) {
      let owner = context.coordinator.id
      view.activate = { [weak view] in
        guard let view else { return }
        if session.webView.superview !== view {
          session.webView.removeFromSuperview()
          view.addSubview(session.webView)
          session.webView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        }
        session.webView.frame = view.bounds
        session.update(
          owner: owner, snapshot: snapshot, scanID: scanID, fixture: fixture, webBaseURL: webBaseURL,
          onPin: onPin, onScan: onScan)
      }
      view.deactivate = { session.detach(owner: owner) }
      if view.window != nil { view.activate?() }
    }
    static func dismantleUIView(_ uiView: GlobeContainer, coordinator: Coordinator) {
      uiView.deactivate?()
      uiView.activate = nil
      uiView.deactivate = nil
    }
  }

  /// Hidden navigation destinations cannot overwrite the visible globe's retained session.
  private final class GlobeContainer: UIView {
    var activate: (() -> Void)?
    var deactivate: (() -> Void)?
    override func didMoveToWindow() {
      super.didMoveToWindow()
      if window == nil { deactivate?() } else { activate?() }
    }

  }
#endif
