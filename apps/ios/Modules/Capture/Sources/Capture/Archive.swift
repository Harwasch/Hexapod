import Foundation

/// USTAR, regular flat files only; bounded memory even for multi-gigabyte HEVC files.
public enum CaptureArchive {
  /// Read only bounded metadata; skip video/depth bytes without loading them into memory.
  /// Full inventory and observation validation remains the pipeline's responsibility.
  public static func readManifest(from archive: URL) throws -> CaptureManifest {
    let input = try FileHandle(forReadingFrom: archive)
    defer { try? input.close() }
    let length = try input.seekToEnd()
    try input.seek(toOffset: 0)
    while try input.offset() + 512 <= length {
      let header = [UInt8](try input.read(upToCount: 512) ?? Data())
      guard header.count == 512, header.contains(where: { $0 != 0 }) else { break }
      func field(_ range: Range<Int>) -> String {
        String(decoding: header[range].prefix(while: { $0 != 0 }), as: UTF8.self)
          .trimmingCharacters(in: .whitespaces)
      }
      let checksum = header.enumerated().reduce(0) {
        $0 + ((148..<156).contains($1.offset) ? 32 : Int($1.element))
      }
      guard field(257..<263) == "ustar", header[156] == 48,
        field(345..<500).isEmpty, CaptureBundleWriter.safeName(field(0..<100)),
        Int(field(148..<156), radix: 8) == checksum,
        let size = UInt64(field(124..<136), radix: 8), size < 8_589_934_592
      else { throw CaptureError.invalidFile }
      let offset = try input.offset()
      guard size <= length - offset else { throw CaptureError.incomplete }
      if field(0..<100) == "manifest.json" {
        guard size <= 64 * 1024 * 1024 else { throw CaptureError.invalidFile }
        let data = try input.read(upToCount: Int(size)) ?? Data()
        guard data.count == Int(size) else { throw CaptureError.incomplete }
        let manifest = try JSONDecoder().decode(CaptureManifest.self, from: data)
        guard manifest.version == 1, manifest.format == "hex-capture" else {
          throw CaptureError.invalidFile
        }
        return manifest
      }
      try input.seek(toOffset: offset + size + (512 - size % 512) % 512)
    }
    throw CaptureError.incomplete
  }

  public static func write(bundle: URL, to destination: URL) throws {
    let manifestURL = bundle.appendingPathComponent("manifest.json")
    let manifest = try JSONDecoder().decode(
      CaptureManifest.self, from: Data(contentsOf: manifestURL))
    guard manifest.version == 1, manifest.format == "hex-capture",
      !FileManager.default.fileExists(atPath: destination.path),
      !destination.standardizedFileURL.path.hasPrefix(bundle.standardizedFileURL.path + "/")
    else { throw CaptureError.invalidFile }
    let names = ["manifest.json"] + manifest.files.map(\.path)
    guard Set(names).count == names.count else { throw CaptureError.invalidFile }
    guard FileManager.default.createFile(atPath: destination.path, contents: nil) else {
      throw CaptureError.invalidFile
    }
    let output = try FileHandle(forWritingTo: destination)
    do {
      for name in names {
        guard CaptureBundleWriter.safeName(name) else { throw CaptureError.invalidFile }
        let url = bundle.appendingPathComponent(name)
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
          throw CaptureError.invalidFile
        }
        let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as! NSNumber)
          .uint64Value
        guard size < 8_589_934_592 else { throw CaptureError.invalidFile }
        if name != "manifest.json" {
          guard manifest.files.first(where: { $0.path == name })?.bytes == size else {
            throw CaptureError.invalidFile
          }
        }
        var header = [UInt8](repeating: 0, count: 512)
        func put(_ text: String, _ offset: Int) {
          for (i, byte) in text.utf8.enumerated() { header[offset + i] = byte }
        }
        func octal(_ n: UInt64, _ width: Int) -> String {
          let digits = String(n, radix: 8)
          return String(repeating: "0", count: width - 1 - digits.count) + digits + "\0"
        }
        put(name, 0)
        put("0000600\0", 100)
        put("0000000\0", 108)
        put("0000000\0", 116)
        put(octal(size, 12), 124)
        put("00000000000\0", 136)
        put("        ", 148)
        header[156] = 48
        put("ustar\0", 257)
        put("00", 263)
        put(octal(UInt64(header.reduce(0) { $0 + Int($1) }), 7) + " ", 148)
        try output.write(contentsOf: Data(header))
        let input = try FileHandle(forReadingFrom: url)
        do {
          var remaining = size
          while remaining > 0 {
            let bytes = try input.read(upToCount: Int(min(remaining, 1024 * 1024))) ?? Data()
            guard !bytes.isEmpty else { throw CaptureError.incomplete }
            try output.write(contentsOf: bytes)
            remaining -= UInt64(bytes.count)
          }
          try input.close()
        } catch {
          try? input.close()
          throw error
        }
        try output.write(contentsOf: Data(repeating: 0, count: Int((512 - size % 512) % 512)))
      }
      try output.write(contentsOf: Data(repeating: 0, count: 1024))
      try output.synchronize()
      try output.close()
    } catch {
      try? output.close()
      try? FileManager.default.removeItem(at: destination)
      throw error
    }
  }
}
