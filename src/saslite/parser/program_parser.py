"""Program parser — uses Lark with combined grammar."""

from __future__ import annotations

from pathlib import Path

from lark import Lark

from saslite.ast.program import ProgramNode
from saslite.parser.transformer import SasTransformer

_GRAMMAR_PATH = Path(__file__).parent / "grammar" / "saslite.lark"


class ProgramParser:
    """Parses expanded SAS source into an AST."""

    def __init__(self) -> None:
        grammar_text = _GRAMMAR_PATH.read_text(encoding="utf-8")
        self._parser = Lark(
            grammar_text,
            parser="earley",
            ambiguity="resolve",
            start="start",
            keep_all_tokens=True,
        )
        self._transformer = SasTransformer()

    def parse(self, source: str) -> ProgramNode:
        """Parse source text into a ProgramNode AST."""
        tree = self._parser.parse(source)
        result = self._transformer.transform(tree)
        if isinstance(result, ProgramNode):
            return result
        return ProgramNode(steps=[result] if result else [])

    def parse_expression(self, text: str) -> object:
        """Parse a single expression."""
        # Wrap in DATA step context for proper parsing
        wrapped = f"DATA _NULL_; x = {text}; RUN;"
        program = self.parse(wrapped)
        if program.steps and hasattr(program.steps[0], "statements"):
            for stmt in program.steps[0].statements:
                if hasattr(stmt, "expr"):
                    return stmt.expr
        return None
