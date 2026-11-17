#if canImport(SwiftUI) && canImport(MapKit) && canImport(UIKit)
  import SwiftUI
  import ARAnswer
  import MapKit
  import EntitySchema
  import SafariServices

  /// Apple UI is unverified on Linux: compile and exercise on iOS 26 before shipping.
  /// Embed in App A's NavigationStack; nil route provides the Home property picker.
  @MainActor
  public struct GlobeView: View {
    private let source: any GlobeDataSource
    private let route: GlobeRoute?
    private let webBaseURL: URL?
    private let onOpenScan: (@MainActor (GlobeRoute) -> Void)?
    @State private var sites: [GlobeSite] = []
    @State private var snapshot: GlobeSnapshot?
    @State private var selectedPin: GlobePin?
    @State private var preview: PreviewRoute?
    @State private var failed = false
    @State private var retry = 0
    @State private var selectedScan: UUID?
    @AppStorage("globeImplementation") private var implementation = GlobeImplementation.defaultValue
      .rawValue
    private let fixture: Bool
    private let renderer: GlobeImplementation?

    public init(
      source: any GlobeDataSource, route: GlobeRoute? = nil, webBaseURL: URL? = nil,
      onOpenScan: (@MainActor (GlobeRoute) -> Void)? = nil, fixture: Bool = false,
      renderer: GlobeImplementation? = nil
    ) {
      self.renderer = renderer
      self.fixture = fixture
      self.source = source
      self.route = route
      self.webBaseURL = webBaseURL
      self.onOpenScan = onOpenScan
    }

    public var body: some View {
      Group {
        if failed {
          Button("Retry", systemImage: "arrow.clockwise") { retry += 1 }
        } else if let snapshot {
          VStack(spacing: 0) {
            if (renderer?.rawValue ?? implementation) == GlobeImplementation.appleMaps.rawValue {
              AppleMapsGlobeView(
                snapshot: snapshot, selectedScan: selectedScan ?? route?.scanID,
                onPin: { selectedPin = $0 }, onScan: { selectedScan = $0 })
            } else {
              CesiumGlobeView(
                snapshot: snapshot, scanID: selectedScan ?? route?.scanID, fixture: fixture,
                onPin: { selectedPin = $0 }, onScan: { selectedScan = $0 })
            }
            if !snapshot.property.epochs.isEmpty {
              Picker("Scan", selection: $selectedScan) {
                Text("Choose scan").tag(nil as UUID?)
                ForEach(snapshot.property.epochs) { scan in
                  Text(scan.observedAt).tag(Optional(scan.captureId ?? scan.id))
                }
              }
              if let selectedScan {
                Button("Open scan") { openScan(selectedScan, siteID: snapshot.property.id) }
                  .disabled(onOpenScan == nil && webBaseURL == nil)
              }
            }
          }
        } else if route == nil {
          List(sites) { site in
            NavigationLink(site.name, value: GlobeRoute(siteID: site.id))
          }
        } else {
          ProgressView()
        }
      }
      .navigationTitle(snapshot?.property.name ?? "Globe")
      .navigationDestination(for: GlobeRoute.self) { next in
        GlobeView(
          source: source, route: next, webBaseURL: webBaseURL, onOpenScan: onOpenScan,
          fixture: fixture, renderer: renderer)
      }
      .toolbar {
        #if DEBUG
          Menu("Debug") {
            Picker("Globe", selection: $implementation) {
              ForEach(GlobeImplementation.allCases, id: \.rawValue) { choice in
                Text(choice.rawValue).tag(choice.rawValue)
              }
            }
          }
        #endif
      }
      .task { _ = CesiumGlobeSession.shared }
      .task(id: LoadKey(route: route, retry: retry)) {
        failed = false
        snapshot = nil
        do {
          if let route {
            let loaded = try await source.snapshot(siteID: route.siteID)
            try Task.checkCancellation()
            snapshot = loaded
          } else {
            let loaded = try await source.sites()
            try Task.checkCancellation()
            sites = loaded
          }
        } catch is CancellationError {} catch { if !Task.isCancelled { failed = true } }
      }
      .sheet(item: $selectedPin) { pin in
        NavigationStack {
          List {
            EntityCard(entity: pin.entity) { selectedPin = nil }
            Text(pin.observedAt)
            if let uncertainty = pin.uncertaintyM { Text("±\(uncertainty, specifier: "%.1f") m") }
            if onOpenScan != nil {
              Button("Scan", systemImage: "viewfinder") { openScan(pin.scanId, siteID: pin.siteId) }
            } else if let webBaseURL {
              NavigationLink("Scan") {
                ScanPreview(url: globeScanPreviewURL(webBaseURL: webBaseURL, siteID: pin.siteId))
              }
            }
          }
          .toolbar { Button("Done") { selectedPin = nil } }
        }
        .presentationDetents([.medium])
      }
      .sheet(item: $preview) { route in ScanPreview(url: route.url) }
    }

    private func openScan(_ scanID: UUID?, siteID: UUID) {
      selectedPin = nil
      if let onOpenScan {
        onOpenScan(
          snapshot?.captureRoute(for: scanID) ?? GlobeRoute(siteID: siteID, scanID: scanID))
      } else if let webBaseURL {
        preview = PreviewRoute(url: globeScanPreviewURL(webBaseURL: webBaseURL, siteID: siteID))
      }
    }
    private struct LoadKey: Hashable {
      let route: GlobeRoute?
      let retry: Int
    }
    private struct PreviewRoute: Identifiable {
      let url: URL
      var id: URL { url }
    }
  }

  @MainActor
  public struct AppleMapsGlobeView: GlobeRenderer {
    public let snapshot: GlobeSnapshot
    let selectedScan: UUID?
    let onPin: (GlobePin) -> Void
    let onScan: (UUID) -> Void
    public var body: some View {
      Map {
        ForEach(Array(snapshot.property.boundary.polygons.enumerated()), id: \.offset) { _, rings in
          MapPolygon(polygon(rings)).foregroundStyle(.green.opacity(0.12)).stroke(
            .green, lineWidth: 2)
        }
        ForEach(snapshot.property.epochs) { scan in
          ForEach(Array(scan.footprint.polygons.enumerated()), id: \.offset) { _, rings in
            MapPolygon(polygon(rings)).foregroundStyle(.cyan.opacity(0.16)).stroke(
              .cyan,
              lineWidth: selectedScan != nil && (scan.captureId ?? scan.id) == selectedScan ? 3 : 1)
          }
          if let point = scan.footprint.polygons.first?.first?.first, point.count >= 2 {
            Annotation(
              "Scan", coordinate: CLLocationCoordinate2D(latitude: point[1], longitude: point[0])
            ) {
              Button {
                onScan(scan.captureId ?? scan.id)
              } label: {
                Image(systemName: "viewfinder").padding(10).background(
                  .regularMaterial, in: Circle())
              }.accessibilityLabel("Scan \(scan.observedAt)")

            }.annotationTitles(.hidden)
          }
        }
        ForEach(snapshot.pins) { pin in
          if let geo = pin.coordinate {
            Annotation(
              pin.entity.label,
              coordinate: CLLocationCoordinate2D(latitude: geo.lat, longitude: geo.lon)
            ) {
              Button {
                onPin(pin)
              } label: {
                Image(systemName: "mappin.circle.fill").font(.title).foregroundStyle(.orange)
              }.accessibilityLabel(pin.entity.label)
            }.annotationTitles(.hidden)
          }
        }
      }
      .mapStyle(.imagery(elevation: .realistic))
      .mapControls {
        MapCompass()
        MapPitchToggle()
        MapScaleView()
      }

    }

    private func polygon(_ rings: [[[Double]]]) -> MKPolygon {
      func ring(_ positions: [[Double]]) -> MKPolygon {
        let coordinates = positions.map {
          CLLocationCoordinate2D(latitude: $0[1], longitude: $0[0])
        }
        return MKPolygon(coordinates: coordinates, count: coordinates.count)
      }
      guard let outer = rings.first else { return MKPolygon() }
      let coordinates = outer.map { CLLocationCoordinate2D(latitude: $0[1], longitude: $0[0]) }
      return MKPolygon(
        coordinates: coordinates, count: coordinates.count,
        interiorPolygons: rings.dropFirst().map(ring))
    }

  }

  private struct ScanPreview: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> SFSafariViewController {
      SFSafariViewController(url: url)
    }
    func updateUIViewController(_ controller: SFSafariViewController, context: Context) {}
  }

  @MainActor
  public struct GlobeDebugView: View {
    public init() {}
    public var body: some View {
      NavigationStack {
        GlobeView(
          source: FakeGlobeDataSource(), route: GlobeRoute(siteID: FakeGlobeDataSource.siteID),
          fixture: true, renderer: .cesium)
      }
    }
  }

  #Preview { GlobeDebugView() }
#endif
