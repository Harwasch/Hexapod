"""Arithmetic interpreter: no eval, imports, attributes, loops, I/O or arbitrary code."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Callable
from typing import Any, cast

from app.schemas.calculations import CalculationIssue, CalculationOutput, CalculationRequest
from app.services.errors import InvalidInputError

METHOD = (
    "Recomputed by the bounded arithmetic-v1 engine from the saved inputs and formulas. "
    "Input values and their stated bases are supplied by the research agent; computation does not "
    "independently verify their accuracy. Assumptions are not measurements. Units are labels: "
    "conversions must be explicit in the formulas. Missing or undefined results remain gaps. "
    "Uses bounded IEEE-754 double-precision arithmetic."
)


class MissingInputError(ValueError):
    pass


def rounded(value: float, digits: float = 0.0) -> float:
    if not digits.is_integer() or abs(digits) > 15:
        raise ValueError("round requires an integer precision between -15 and 15")
    return round(value, int(digits))


FUNCTIONS: dict[str, tuple[Callable[..., float], int, int]] = {
    "abs": (abs, 1, 1),
    "sqrt": (math.sqrt, 1, 1),
    "log": (math.log, 1, 2),
    "log10": (math.log10, 1, 1),
    "exp": (math.exp, 1, 1),
    "floor": (math.floor, 1, 1),
    "ceil": (math.ceil, 1, 1),
    "round": (rounded, 1, 2),
    "min": (lambda *args: min(args), 1, 20),
    "max": (lambda *args: max(args), 1, 20),
    "sum": (lambda *args: math.fsum(args), 1, 20),
}


def finite(value: float) -> float:
    if not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("The result exceeds the finite arithmetic range")
    return value


class Expression:
    def __init__(self, expression: str, names: set[str]) -> None:
        try:
            self.tree = ast.parse(expression, mode="eval").body
        except (SyntaxError, ValueError) as error:
            raise InvalidInputError("Use a valid arithmetic expression.") from error
        if len(list(ast.walk(self.tree))) > 100:
            raise InvalidInputError("A formula supports at most 100 expression nodes.")
        self.validate(self.tree, names)

    def validate(self, node: ast.AST, names: set[str]) -> None:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                finite(float(cast(int | float, node.value)))
            except (ValueError, OverflowError) as error:
                raise InvalidInputError(
                    "Formula constants must be finite and at most 1e100."
                ) from error
        elif isinstance(node, ast.Name) and node.id in names:
            return
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self.validate(node.operand, names)
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        ):
            self.validate(node.left, names)
            self.validate(node.right, names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
        ):
            _, minimum, maximum = FUNCTIONS[node.func.id]
            if node.keywords or not minimum <= len(node.args) <= maximum:
                raise InvalidInputError(
                    f"{node.func.id} accepts {minimum}-{maximum} positional arguments."
                )
            for argument in node.args:
                self.validate(argument, names)
        else:
            raise InvalidInputError(
                "Only numbers, known inputs/earlier results, arithmetic and registered functions are allowed."
            )

    def evaluate(self, values: dict[str, float | None]) -> float:
        def visit(node: ast.AST) -> float:
            if isinstance(node, ast.Constant):
                return float(cast(int | float, node.value))
            if isinstance(node, ast.Name):
                value = values[node.id]
                if value is None:
                    raise MissingInputError(f"Missing input or prior result: {node.id}")
                return value
            if isinstance(node, ast.UnaryOp):
                value = visit(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return finite(
                    float(FUNCTIONS[node.func.id][0](*[visit(argument) for argument in node.args]))
                )
            if isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                elif isinstance(node.op, ast.Div):
                    result = left / right
                elif isinstance(node.op, ast.FloorDiv):
                    result = left // right
                elif isinstance(node.op, ast.Mod):
                    result = left % right
                else:
                    if abs(right) > 100:
                        raise ValueError("Exponents must be between -100 and 100")
                    result = math.pow(left, right)
                return finite(result)
            raise InvalidInputError("Unsupported formula node.")

        return visit(self.tree)


def calculate(request: CalculationRequest) -> CalculationOutput:
    names = {item.name for item in request.inputs}
    if any(name in FUNCTIONS for name in [*names, *[item.name for item in request.formulas]]):
        raise InvalidInputError(
            "Input/result names must not shadow registered arithmetic functions."
        )
    expressions = []
    for formula in request.formulas:
        expressions.append(Expression(formula.expression, names))
        names.add(formula.name)
    rows: list[dict[str, float | None]] = []
    issues = []
    for index in range(len(request.row_labels)):
        values = {
            item.name: item.values[0 if len(item.values) == 1 else index] for item in request.inputs
        }
        row: dict[str, float | None] = {}
        for formula, expression in zip(request.formulas, expressions, strict=True):
            try:
                result: float | None = expression.evaluate(values)
            except (ValueError, ArithmeticError) as error:
                result = None
                issues.append(
                    CalculationIssue(
                        row=index,
                        column=formula.name,
                        kind="missing" if isinstance(error, MissingInputError) else "undefined",
                        message=str(error)[:300] or "Undefined arithmetic result",
                    )
                )
            values[formula.name] = result
            row[formula.name] = result
        rows.append(row)
    serialized = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return CalculationOutput(
        kind="calculation",
        engine_version="arithmetic-v1",
        request=request,
        request_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        rows=rows,
        issues=issues,
    )


def page(output: CalculationOutput, offset: int, count: int) -> dict[str, Any]:
    if offset < 0 or offset >= len(output.rows) or not 1 <= count <= 50:
        raise InvalidInputError("The requested page is outside the saved calculation.")
    end = min(offset + count, len(output.rows))
    return {
        "requestSha256": output.request_sha256,
        "engineVersion": output.engine_version,
        "purpose": output.request.purpose,
        "limitations": output.request.limitations,
        "labels": output.request.row_labels[offset:end],
        "rows": output.rows[offset:end],
        "inputs": [
            {
                **item.model_dump(mode="json", by_alias=True),
                "values": item.values if len(item.values) == 1 else item.values[offset:end],
            }
            for item in output.request.inputs
        ],
        "formulas": [
            item.model_dump(mode="json", by_alias=True) for item in output.request.formulas
        ],
        "issues": [
            item.model_dump(mode="json", by_alias=True)
            for item in output.issues
            if offset <= item.row < end
        ],
        "offset": offset,
        "total": len(output.rows),
        "nextOffset": end if end < len(output.rows) else None,
    }
