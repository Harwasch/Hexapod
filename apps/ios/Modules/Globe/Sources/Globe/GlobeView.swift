#if canImport(SwiftUI) && canImport(MapKit) && canImport(UIKit)
  import SwiftUI
  import ARAnswer
  import MapKit
  import EntitySchema
  import UIKit

  /// Apple UI is unverified on Linux: compile and exercise on iOS 26 before shipping.
  /// Embed in App A's NavigationStack; nil route provides the Home property picker.
  @MainActor
  public struct GlobeView: View {
    @Environment(\.dismiss) private var dismiss
    private let source: any GlobeDataSource
    private let route: GlobeRoute?
    private let webBaseURL: URL?
    private let onOpenScan: (@MainActor (GlobeRoute) -> Void)?
    @State private var sites: [GlobeSite] = []
    @State private var snapshot: GlobeSnapshot?
    @State private var selectedPin: GlobePin?
    @State private var failed = false
    @State private var retry = 0
    @State private var selectedScan: UUID?
    @State private var selectedSite: GlobeRoute?
    @State private var loadedSites = false
    @State private var scanPending = false
    private var effectiveRoute: GlobeRoute? { route ?? selectedSite }
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
          VStack(spacing: 12) {
            Label("Globe unavailable", systemImage: "wifi.exclamationmark")
            Button("Retry", systemImage: "arrow.clockwise") { retry += 1 }.frame(minHeight: 44)
          }
        } else if scanPending {
          VStack(spacing: 12) {
            Label("Scan pending", systemImage: "clock")
            Button("Refresh") { retry += 1 }.frame(minHeight: 44)
          }
        } else if let snapshot {
          ZStack {
            if (renderer?.rawValue ?? implementation) == GlobeImplementation.appleMaps.rawValue {
              AppleMapsGlobeView(
                snapshot: snapshot, selectedScan: selectedScan ?? effectiveRoute?.scanID,
                onPin: { selectedPin = $0 }, onScan: { selectedScan = $0 }
              )
              .ignoresSafeArea()
            } else {
              CesiumGlobeView(
                snapshot: snapshot, scanID: selectedScan ?? effectiveRoute?.scanID,
                fixture: fixture, webBaseURL: webBaseURL,
                onPin: { selectedPin = $0 }, onScan: { selectedScan = $0 }
              )
              .ignoresSafeArea()
            }
          }
          .overlay(alignment: .bottom) {
            if snapshot.property.epochs.isEmpty {
              Label("No scans yet", systemImage: "viewfinder")
                .padding(12).background(.regularMaterial, in: Capsule())
                .padding(.bottom, 64).accessibilityIdentifier("globe.empty")
            }
          }
          .overlay(alignment: .topTrailing) {
            Menu {
              if sites.count > 1 && route == nil {
                Picker(
                  "Site",
                  selection: Binding(
                    get: { snapshot.property.id },
                    set: {
                      selectedSite = GlobeRoute(siteID: $0)
                      selectedScan = nil
                    }
                  )
                ) {
                  ForEach(sites) { site in Text(site.name).tag(site.id) }
                }.accessibilityIdentifier("globe.sitePicker")
              }
              if !snapshot.property.epochs.isEmpty {
                Picker("Scan", selection: $selectedScan) {
                  ForEach(snapshot.property.epochs) { scan in
                    Text(globeScanDate(scan.observedAt)).tag(Optional(scan.captureId ?? scan.id))
                      .accessibilityIdentifier(
                        "globe.scan.\((scan.captureId ?? scan.id).uuidString.lowercased())")
                  }
                }
              }
              if let selectedScan {
                Button("Open scan", systemImage: "viewfinder") {
                  openScan(selectedScan, siteID: snapshot.property.id)
                }
              }
            } label: {
              Label(snapshot.property.name, systemImage: "globe.americas")
                .font(.subheadline.weight(.medium)).lineLimit(1).frame(maxWidth: 220)
                .padding(.horizontal, 14).frame(minHeight: 44)
                .background(.regularMaterial, in: Capsule())
            }.accessibilityIdentifier("globe.scanPicker")
              .padding(.trailing, 16).padding(.top, 8)
          }
        } else if effectiveRoute == nil && loadedSites {
          if sites.isEmpty {
            Label("No sites yet", systemImage: "globe.americas")
              .accessibilityIdentifier("globe.noSites")
          } else {
            List(sites) { site in
              Button(site.name) { selectedSite = GlobeRoute(siteID: site.id) }
                .frame(minHeight: 44).accessibilityIdentifier("globe.site.\(site.id)")
            }.scrollContentBackground(.hidden)
          }
        } else {
          ProgressView("Connecting…")
        }
      }
      .overlay(alignment: .topLeading) {
        Button {
          dismiss()
        } label: {
          Image(systemName: "chevron.left").frame(width: 44, height: 44)
            .background(.regularMaterial, in: Circle())
        }.accessibilityLabel("Back").accessibilityIdentifier("globe.back")
          .padding(.leading, 16).padding(.top, 8)
      }
      .navigationBarBackButtonHidden()
      .toolbar(.hidden, for: .navigationBar)
      .task { CesiumGlobeSession.shared.configure(webBaseURL: webBaseURL, fixture: fixture) }
      .task(id: LoadKey(route: effectiveRoute, retry: retry)) {
        failed = false
        scanPending = false
        snapshot = nil
        do {
          if let route = effectiveRoute {
            let loaded = try await source.snapshot(siteID: route.siteID)
            try Task.checkCancellation()
            if let scanID = route.scanID,
              !loaded.property.epochs.contains(where: { ($0.captureId ?? $0.id) == scanID })
            {
              scanPending = true
              return
            }
            snapshot = loaded.resolvingAssetURLs(relativeTo: webBaseURL)
            selectedScan =
              route.scanID ?? loaded.property.epochs.first.map { $0.captureId ?? $0.id }
          } else {
            let loaded = try await source.sites()
            try Task.checkCancellation()
            sites = loaded
            loadedSites = true
            selectedSite = globeInitialRoute(sites: loaded)
          }
        } catch is CancellationError {} catch { if !Task.isCancelled { failed = true } }
      }
      .sheet(item: $selectedPin) { pin in
        NavigationStack {
          VStack(spacing: 12) {
            EntityCard(entity: pin.entity) { selectedPin = nil }
              .accessibilityIdentifier("globe.entityCard")
            Text(globeScanDate(pin.observedAt))
            if let uncertainty = pin.uncertaintyM { Text("±\(uncertainty, specifier: "%.1f") m") }
            Button("Scan", systemImage: "viewfinder") { openScan(pin.scanId, siteID: pin.siteId) }
          }
          .toolbar { Button("Done") { selectedPin = nil } }
        }
        .presentationDetents([.medium])
      }
    }

    private func openScan(_ scanID: UUID?, siteID: UUID) {
      selectedPin = nil
      if let onOpenScan {
        onOpenScan(
          snapshot?.captureRoute(for: scanID) ?? GlobeRoute(siteID: siteID, scanID: scanID))
      } else if let target = snapshot?.captureRoute(for: scanID).scanID {
        selectedScan = target
        implementation = GlobeImplementation.cesium.rawValue
        CesiumGlobeSession.shared.focus(scanID: target)
      }
    }
    private struct LoadKey: Hashable {
      let route: GlobeRoute?
      let retry: Int
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
