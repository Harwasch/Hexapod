#if canImport(WebKit) && os(iOS)
  import Foundation
  import Globe
  import Observation
  import SwiftUI
  import WebKit
  import UIKit

  /// Reuse the existing globe-embed renderer and public bridge, with a screen-owned
  /// WebView. A second host must not reparent the globe's shared WebView. The
  /// documented data-scan-state signal gates the fade until the tileset is loaded.
  @MainActor @Observable final class TrainingCloudController: NSObject, WKNavigationDelegate {
    let webView: WKWebView
    var ready = false
    var failed = false
    private let route: GlobeRoute
    private let origin: GlobeOrigin
    private let apiRoot: URL?
    private var snapshot: GlobeSnapshot?
    private var pageReady = false
    private var generation = UUID()
    var onReady: (() -> Void)?

    init(route: GlobeRoute, configuration: ServiceConfiguration) {
      self.route = route
      origin = GlobeOrigin(webBaseURL: configuration.webBaseURL ?? GlobeOrigin.deployed)
      apiRoot = configuration.apiRoot
      let config = WKWebViewConfiguration()
      config.websiteDataStore = .nonPersistent()
      webView = WKWebView(frame: .zero, configuration: config)
      super.init()
      webView.navigationDelegate = self
      webView.isOpaque = false
      webView.scrollView.isScrollEnabled = false
      config.userContentController.add(TrainingCloudMessages(owner: self), name: "trainingCloud")
      config.userContentController.add(TrainingCloudMessages(owner: self), name: "globe")
      NotificationCenter.default.addObserver(self, selector: #selector(memoryWarning),
        name: UIApplication.didReceiveMemoryWarningNotification, object: nil)
      // This observes the existing embed contract; no rendering code is duplicated.
      let script = """
        (() => {
          const emit = value => window.webkit.messageHandlers.trainingCloud.postMessage(value);
          const check = () => {
            const state = document.body.dataset.scanState;
            if (state === 'ready' || state === 'failed') emit(state);
          };
          document.addEventListener('dblclick', () => emit('focus'));
          new MutationObserver(check).observe(document.body, { attributes: true, attributeFilter: ['data-scan-state'] });
          check();
        })();
        """
      config.userContentController.addUserScript(WKUserScript(source: script,
        injectionTime: .atDocumentEnd, forMainFrameOnly: true))
    }
    deinit { NotificationCenter.default.removeObserver(self) }
    @objc private func memoryWarning() {
      webView.callAsyncJavaScript(
        "if (window.globeBridge) window.globeBridge.receive({version: 1, type: 'memoryWarning'});",
        arguments: [:], in: nil, in: .page) { _ in }
    }
    func load() async {
      let requestID = UUID()
      generation = requestID
      ready = false; failed = false; pageReady = false; snapshot = nil
      webView.load(URLRequest(url: origin.page))
      guard let apiRoot else { failed = true; return }
      do {
        let value = try await APIGlobeDataSource(baseURL: apiRoot).snapshot(siteID: route.siteID)
        try Task.checkCancellation()
        guard generation == requestID else { return }
        snapshot = value.resolvingAssetURLs(relativeTo: origin.page)
        sendSnapshotIfReady()
      } catch { if generation == requestID { failed = true } }
    }
    func receive(_ message: WKScriptMessage) {
      let sender = message.frameInfo.securityOrigin
      guard message.webView === webView, message.frameInfo.isMainFrame,
        origin.allowsMessage(message.frameInfo.request.url),
        origin.allowsSecurityOrigin(scheme: sender.`protocol`, host: sender.host, port: sender.port)
      else { return }
      if message.name == "globe" {
        guard let data = try? JSONSerialization.data(withJSONObject: message.body),
          let event = try? JSONDecoder().decode(BridgeMessage.self, from: data) else { return }
        switch event {
        case .ready:
          guard !pageReady else { return }
          pageReady = true
          sendSnapshotIfReady()
        case .failure: failed = true
        default: break
        }
        return
      }
      guard let value = message.body as? String else { return }
      switch value {
      case "ready":
        guard snapshot != nil, !ready else { return }
        ready = true; failed = false; onReady?()
      case "failed": failed = true
      case "focus": focus()
      default: break
      }
    }
    private func sendSnapshotIfReady() {
      guard pageReady, let snapshot, let scan = route.scanID else { return }
      let bridge = snapshot.bridgeSnapshot()
      let scanID = scan.uuidString.lowercased()
      guard bridge.scans.contains(where: { $0.id == scanID && $0.tilesetURL != nil }) else {
        failed = true
        return
      }
      let messages: [BridgeMessage] = [.snapshot(snapshot: bridge), .selectScan(scanId: scanID), .flyToScan(scanId: scanID)]
      var payloads: [String] = []
      for message in messages {
        guard let data = try? JSONEncoder().encode(message), let payload = String(data: data, encoding: .utf8) else { return }
        payloads.append(payload)
      }
      webView.callAsyncJavaScript(
        "for (const payload of payloads) window.globeBridge.receive(JSON.parse(payload));",
        arguments: ["payloads": payloads], in: nil, in: .page
      ) { [weak self] result in
        if case .failure = result { self?.failed = true }
      }
    }
    func focus() {
      guard let scanID = route.scanID?.uuidString.lowercased(), ready else { return }
      webView.callAsyncJavaScript(
        "window.globeBridge.receive({version: 1, type: 'flyToScan', scanId: scanID});",
        arguments: ["scanID": scanID], in: nil, in: .page
      ) { _ in }
    }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: any Error) { failed = true }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: any Error) { failed = true }
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) { failed = true }
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
      decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void) {
      let allowed = navigationAction.targetFrame?.isMainFrame == true && origin.allowsNavigation(navigationAction.request.url)
      decisionHandler(allowed ? .allow : .cancel)
    }
  }
  @MainActor private final class TrainingCloudMessages: NSObject, WKScriptMessageHandler {
    weak var owner: TrainingCloudController?
    init(owner: TrainingCloudController) { self.owner = owner }
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
      owner?.receive(message)
    }
  }
  @MainActor struct TrainingCloudPreview: UIViewRepresentable {
    let controller: TrainingCloudController
    func makeUIView(context: Context) -> WKWebView { controller.webView }
    func updateUIView(_ view: WKWebView, context: Context) {}
    static func dismantleUIView(_ view: WKWebView, coordinator: ()) { view.stopLoading() }
  }
#endif
