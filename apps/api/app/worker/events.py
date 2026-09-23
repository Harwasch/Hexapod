"""The line protocol between the supervisor and the process running the recipe.

One JSON document per line on the child's stdout. It is deliberately dull: the child
never opens a database connection, so everything the supervisor writes to `job_step` and
`artifacts` arrives through here, as it happens, and a child that dies takes nothing with
it but its own process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: Sent before a stage runs, so the step row exists while the stage is still running.
STAGE_STARTED = "stage-started"
#: Sent after a stage's outputs have been verified and its StepResult written.
STAGE_FINISHED = "stage-finished"
#: Sent for a stage a retry skipped, whose previous result stands.
STAGE_SKIPPED = "stage-skipped"
#: Sent when a stage raised. `error` is the message a person reads in the panel.
STAGE_FAILED = "stage-failed"
#: The recipe ran to the end.
RUN_FINISHED = "run-finished"
#: The run stopped. `stage_id` is set when a stage was to blame and empty when the recipe
#: itself would not resolve — the supervisor retries the first and never the second.
RUN_FAILED = "run-failed"


@dataclass(frozen=True)
class Event:
    kind: str
    stage_id: str = ""
    ordinal: int = -1
    impl: str = ""
    attempt: int = 1
    error: str = ""
    error_type: str = ""
    #: `StepResult.to_dict()` for a finished or skipped stage.
    step: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "stageId": self.stage_id,
                "ordinal": self.ordinal,
                "impl": self.impl,
                "attempt": self.attempt,
                "error": self.error,
                "errorType": self.error_type,
                "step": self.step,
            },
            sort_keys=True,
        )

    @staticmethod
    def from_json(line: str) -> Event | None:
        """Parse one line, or None if it is not one of ours.

        A child's stdout can pick up a warning from a library that has nothing to do with
        us; that is not a reason to lose the run.
        """
        try:
            document = json.loads(line)
        except ValueError:
            return None
        if not isinstance(document, dict) or "kind" not in document:
            return None
        step = document.get("step")
        return Event(
            kind=str(document["kind"]),
            stage_id=str(document.get("stageId", "")),
            ordinal=int(document.get("ordinal", -1)),
            impl=str(document.get("impl", "")),
            attempt=int(document.get("attempt", 1)),
            error=str(document.get("error", "")),
            error_type=str(document.get("errorType", "")),
            step=step if isinstance(step, dict) else {},
        )
