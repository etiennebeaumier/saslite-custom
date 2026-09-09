"""Step dispatcher — routes AST nodes to the appropriate executor."""

from __future__ import annotations

from typing import Any

from saslite.ast.program import ProgramNode, LibnameNode, FilenameNode, OptionsNode
from saslite.ast.data_step import DataStepNode
from saslite.ast.sql import ProcSqlNode
from saslite.ast.proc import ProcNode
from saslite.session.session import Session
from saslite.runtime.execution_result import StepResult, RunSummary
from saslite.diagnostics.reporter import Reporter


class Dispatcher:
    """Dispatches AST steps to the correct executor."""

    def __init__(self, session: Session, reporter: Reporter) -> None:
        self.session = session
        self.reporter = reporter
        self._proc_handlers: dict[str, Any] = {}

    def register_proc(self, name: str, handler: Any) -> None:
        self._proc_handlers[name.upper()] = handler

    def run(self, program: ProgramNode) -> RunSummary:
        """Execute all steps in a program."""
        summary = RunSummary()

        for step in program.steps:
            if step is None:
                continue

            result = self._dispatch_step(step)
            summary.add_step(result)

            if result.error:
                self.reporter.error(result.error)

            for msg in result.notes:
                self.reporter.note(msg)
            for msg in result.warnings:
                self.reporter.warning(msg)

        return summary

    def _dispatch_step(self, step: Any) -> StepResult:
        """Dispatch a single step to its executor."""
        if isinstance(step, DataStepNode):
            from saslite.executor.data_step.executor import DataStepExecutor
            executor = DataStepExecutor(self.session, self.reporter)
            return executor.run(step)

        if isinstance(step, ProcSqlNode):
            from saslite.executor.sql.executor import SqlExecutor
            executor = SqlExecutor(self.session, self.reporter)
            return executor.run(step)

        if isinstance(step, ProcNode):
            handler = self._proc_handlers.get(step.proc_name.upper())
            if handler is None:
                return StepResult(
                    success=False,
                    error=f"PROC {step.proc_name} is not implemented",
                )
            return handler(step)

        if isinstance(step, LibnameNode):
            from saslite.executor.libname import handle_libname
            return handle_libname(step, self.session, self.reporter)

        if isinstance(step, OptionsNode):
            for name, value in step.options.items():
                self.session.set_option(name, value)
            return StepResult(
                success=True,
                notes=[f"Options updated: {', '.join(sorted(step.options))}"] if step.options else [],
            )

        if isinstance(step, FilenameNode):
            # Store fileref → filepath mapping in session
            self.session.set_macro_var(f"_FILEREF_{step.fileref.upper()}", step.filepath)
            return StepResult(
                success=True,
                notes=[f"Filename {step.fileref} assigned to {step.filepath}"],
            )

        return StepResult(success=False, error=f"Unknown step type: {type(step).__name__}")
