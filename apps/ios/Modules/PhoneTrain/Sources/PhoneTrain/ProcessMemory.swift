#if canImport(Msplat) && canImport(Darwin)
  import Darwin

  /// Independent task_vm_info sampling, matching the reference benchmark's native
  /// phys_footprint metric. A failed Mach query stays nil, never a false zero.
  enum ProcessMemory {
    static func physicalFootprint() -> UInt64? {
      var info = task_vm_info_data_t()
      var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
      let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
          task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
      }
      return status == KERN_SUCCESS ? info.phys_footprint : nil
    }
  }
#endif
