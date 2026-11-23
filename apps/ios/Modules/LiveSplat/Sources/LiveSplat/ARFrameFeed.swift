#if canImport(ARKit) && os(iOS)
  @preconcurrency import ARKit
  import Foundation
  import UIKit

  /// Immutable snapshot retained in a single replaceable mailbox. No pixel copies.
  final class PreviewFrame: @unchecked Sendable {
    let frame: ARFrame
    init(_ frame: ARFrame) { self.frame = frame }
  }
  struct PreviewInput: Sendable {
    var frame: PreviewFrame?
    var active = false
    var enabled = true
    var generation: UInt64 = 0
    var orientation: UIInterfaceOrientation = .portrait
    var viewport = CGSize(width: 1, height: 1)
  }
#endif
