import EntitySchema
import Foundation

public enum FastTool: String, Codable, CaseIterable, Sendable {
  case measure, count, identifyOnDevice, show
}

public struct AnswerPlan: Equatable, Sendable {
  public var subject: String
  public var tools: [FastTool]
  public init(subject: String, tools: [FastTool]) {
    let noun = subject.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    self.subject = ["", "it", "this", "that", "these", "those"].contains(noun) ? "subject" : noun
    self.tools = tools.isEmpty ? [.show] : Array(Set(tools)).sorted { $0.rawValue < $1.rawValue }
  }
}

public protocol Planning: Sendable {
  func plan(_ transcript: String) async throws -> AnswerPlan
}

public struct RulePlanner: Planning {
  public init() {}
  public func plan(_ transcript: String) async throws -> AnswerPlan {
    let words = transcript.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
    let set = Set(words)
    var tools: [FastTool] = []
    if !set.isDisjoint(with: [
      "measure", "height", "tall", "wide", "width", "size", "long", "length",
    ]) {
      tools.append(.measure)
    }
    if !set.isDisjoint(with: ["count", "many", "number"]) { tools.append(.count) }
    if !set.isDisjoint(with: ["identify", "what", "which", "species"]) {
      tools.append(.identifyOnDevice)
    }
    let stop: Set<String> = [
      "measure", "height", "tall", "wide", "width", "size", "long", "length",
      "count", "many", "number", "identify", "what", "which", "species", "show", "me", "the",
      "a", "an", "of", "and", "is", "are", "this", "that", "these", "those", "how", "please", "it",
      "its", "them", "there", "here", "can", "could", "would", "you", "tell", "about",
    ]
    let subject = words.filter { !stop.contains($0) }.joined(separator: " ")
    return AnswerPlan(subject: subject.isEmpty ? "subject" : subject, tools: tools)
  }
}

public protocol FastAnswering: Sendable {
  func answer(_ request: AskRequest, plan: AnswerPlan) async throws -> AsyncThrowingStream<
    AskEvent, Error
  >
}

public struct OfflineDeepAnswer: DeepAnswering {
  public init() {}
  public func ask(_ request: AskRequest) -> AsyncThrowingStream<AskEvent, Error> {
    AsyncThrowingStream {
      $0.yield(.done)
      $0.finish()
    }
  }
}

/// The same matching rule is shared with presentation; Ask entities have no identity field.
public enum EntityMatch {
  public static func matches(_ a: Entity, _ b: Entity, radius: Double = 0.35) -> Bool {
    let pa = a.anchor.world
    let pb = b.anchor.world
    let distance = sqrt(pow(pa.x - pb.x, 2) + pow(pa.y - pb.y, 2) + pow(pa.z - pb.z, 2))
    return a.label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      == b.label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      && distance <= radius
  }
}

public enum AnswerPath: Sendable { case fast, deep }

/// Pure reducer. Deep changes are buffered until the fast stream ends, then override it.
public struct AnswerMerge: Sendable {
  public private(set) var entities: [Entity] = []
  private var deepOwned: Set<Int> = []
  private var fastFinished = false
  private var deepFinished = false
  private var pending: [AskEvent] = []
  private var completed = false
  public init() {}

  public mutating func receive(_ event: AskEvent, from path: AnswerPath) -> [AskEvent] {
    guard !completed else { return [] }
    if event == .done { return finish(path) }
    if path == .deep && !fastFinished {
      pending.append(event)
      return []
    }
    return apply(event, from: path)
  }

  public mutating func finish(_ path: AnswerPath) -> [AskEvent] {
    guard !completed else { return [] }
    var output: [AskEvent] = []
    if path == .fast {
      fastFinished = true
      for event in pending { output += apply(event, from: .deep) }
      pending.removeAll()
    } else {
      deepFinished = true
    }
    if fastFinished && deepFinished {
      completed = true
      output.append(.done)
    }
    return output
  }

  private mutating func apply(_ event: AskEvent, from path: AnswerPath) -> [AskEvent] {
    guard case .entity(let entity) = event else { return [event] }
    if let index = entities.firstIndex(where: { EntityMatch.matches($0, entity) }) {
      if path == .fast && deepOwned.contains(index) { return [] }
      if path == .deep { deepOwned.insert(index) }
      guard entities[index] != entity else { return [] }
      entities[index] = entity
    } else {
      entities.append(entity)
      if path == .deep { deepOwned.insert(entities.count - 1) }
    }
    return [event]
  }
}

/// Owns one question. Cancellation finishes the old stream without a stale done or answer.
public actor AnswerRouter {
  private let planner: any Planning
  private let fast: any FastAnswering
  private let deep: any DeepAnswering
  private var active: Task<Void, Never>?
  private var continuation: AsyncStream<AskEvent>.Continuation?
  private var generation = UUID()
  private var merge = AnswerMerge()

  public init(
    planner: any Planning = RulePlanner(), fast: any FastAnswering,
    deep: any DeepAnswering = OfflineDeepAnswer()
  ) {
    self.planner = planner
    self.fast = fast
    self.deep = deep
  }

  public func cancel() {
    generation = UUID()
    active?.cancel()
    active = nil
    continuation?.finish()
    continuation = nil
  }

  public func ask(_ request: AskRequest) -> AsyncStream<AskEvent> {
    cancel()
    let id = generation
    merge = AnswerMerge()
    let pair = AsyncStream<AskEvent>.makeStream()
    continuation = pair.continuation
    pair.continuation.onTermination = { [weak self] _ in
      Task { await self?.cancel(ifCurrent: id) }
    }
    let planner = planner
    let fast = fast
    let deep = deep
    active = Task { [weak self] in
      guard let owner = self else { return }
      await withTaskGroup(of: Void.self) { group in
        group.addTask {
          do {
            let plan: AnswerPlan
            do { plan = try await planner.plan(request.question) } catch {
              plan = try await RulePlanner().plan(request.question)
            }
            try Task.checkCancellation()
            let stream = try await fast.answer(request, plan: plan)
            for try await event in stream {
              try Task.checkCancellation()
              if event == .done { break }
              await owner.accept(event, from: .fast, id: id)
            }
          } catch {
            if !Task.isCancelled {
              await owner.accept(.status("Fast answer unavailable"), from: .fast, id: id)
            }
          }
          await owner.finish(.fast, id: id)
        }
        group.addTask {
          do {
            for try await event in deep.ask(request) {
              try Task.checkCancellation()
              if event == .done { break }
              await owner.accept(event, from: .deep, id: id)
            }
          } catch {
            if !Task.isCancelled {
              await owner.accept(.status("Deep answer unavailable"), from: .deep, id: id)
            }
          }
          await owner.finish(.deep, id: id)
        }
        await group.waitForAll()
      }
    }
    return pair.stream
  }

  private func cancel(ifCurrent id: UUID) { if id == generation { cancel() } }
  private func accept(_ event: AskEvent, from path: AnswerPath, id: UUID) {
    guard id == generation, !Task.isCancelled else { return }
    emit(merge.receive(event, from: path))
  }
  private func finish(_ path: AnswerPath, id: UUID) {
    guard id == generation, !Task.isCancelled else { return }
    emit(merge.finish(path))
  }
  private func emit(_ events: [AskEvent]) {
    for event in events {
      continuation?.yield(event)
      if event == .done {
        continuation?.finish()
        continuation = nil
        active = nil
      }
    }
  }
}
