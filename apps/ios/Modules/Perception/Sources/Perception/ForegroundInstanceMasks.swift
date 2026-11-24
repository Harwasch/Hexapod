/// Decodes Vision's instance-ID image in one pass, rather than scanning it once per object.
public enum ForegroundInstanceMasks {
  public static func decode(width: Int, height: Int, labels: [UInt8], instances: [Int]) throws
    -> [SubjectMask]
  {
    guard width > 0, height > 0, labels.count / width == height, labels.count % width == 0,
      Set(instances).count == instances.count, instances.allSatisfy({ $0 > 0 && $0 <= 255 })
    else { throw PerceptionError.invalidGeometry }
    var lookup = Array(repeating: -1, count: 256)
    for (index, id) in instances.enumerated() { lookup[id] = index }
    var planes = instances.map { _ in Array(repeating: UInt8(0), count: labels.count) }
    for (pixel, id) in labels.enumerated() {
      let index = lookup[Int(id)]
      if index >= 0 { planes[index][pixel] = 255 }
    }
    return planes.map { SubjectMask(width: width, height: height, pixels: $0) }
  }
}
