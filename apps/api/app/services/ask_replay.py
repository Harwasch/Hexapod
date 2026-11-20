"""Recorded synthetic AskEvent streams; the route still owns ledger commit and done."""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from app.schemas.ask import AskEvent, AskRequest, Entity, SpeechEvent, StatusEvent
from app.services.ask_entities import EVENT
from app.services.errors import InvalidInputError
from app.services.property_memory import MemoryQuery, MemoryRepository

RECORDINGS = Path(__file__).resolve().parents[1] / "fixtures/ask"
QUESTIONS = {
    "how tall is the tree?": "yard-small-height.sse",
    "how many trees?": "yard-small-count.sse",
}


def recorded_events(question: str) -> list[AskEvent]:
    filename = QUESTIONS.get(question.strip().casefold())
    if filename is None:
        raise InvalidInputError("replay question must be: " + " or ".join(QUESTIONS))
    events = [
        EVENT.validate_json(line.removeprefix("data: "))
        for line in (RECORDINGS / filename).read_text().splitlines()
        if line.startswith("data: ")
    ]
    if not events or events[-1].type != "done" or any(e.type == "done" for e in events[:-1]):
        raise ValueError("recording must end with exactly one done")
    return events


class ReplayAnswerer:
    def __init__(self, memory: MemoryRepository | None = None) -> None:
        self.memory = memory

    async def answer(
        self, request: AskRequest, previous: list[Entity], bundle_url: str | None
    ) -> AsyncIterator[AskEvent]:
        if request.scope == "property":
            if self.memory is None or request.site_id is None:
                raise InvalidInputError("property replay requires memory")
            yield StatusEvent(text="Looking")
            question = request.question.casefold()
            label = "shed" if "shed" in question else "tree" if "tree" in question else None
            observations = self.memory.entities(
                uuid.UUID(request.site_id), MemoryQuery(label=label)
            )
            yield SpeechEvent(
                text=f"Simulated replay: {len(observations)} saved observation(s) on this property page."
            )
            return
        for event in recorded_events(request.question)[:-1]:
            yield event
