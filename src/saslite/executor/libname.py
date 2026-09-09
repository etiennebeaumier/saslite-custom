"""LIBNAME statement executor."""

from __future__ import annotations

from pathlib import Path

from saslite.ast.program import LibnameNode
from saslite.session.session import Session
from saslite.storage.csv_backend import CsvBackend
from saslite.storage.memory import MemoryBackend
from saslite.storage.sas_backend import SasBackend
from saslite.runtime.execution_result import StepResult
from saslite.diagnostics.reporter import Reporter


def handle_libname(node: LibnameNode, session: Session, reporter: Reporter) -> StepResult:
    """Execute a LIBNAME statement."""
    libref = node.libref.upper()

    # LIBNAME libref;  — clear/unassign
    if not node.path and not node.engine:
        if libref in session.storage._backends and libref != "WORK":
            del session.storage._backends[libref]
            return StepResult(success=True, notes=[f"Library {libref} has been cleared"])
        return StepResult(success=True, notes=[f"Library {libref} was not assigned"])

    path = node.path
    override = session.get_option('LIBRARY_OVERRIDES', {}).get(libref)
    if override:
        path = override

    # Determine engine
    engine = node.engine.upper() if node.engine else ""
    if override and engine in ('', 'SAS', 'V9', 'BASE'):
        engine = 'SAS'

    if path and engine in ("", "SAS", "XPORT", "V9", "BASE"):
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(session.get_option('SOURCE_DIR', Path.cwd())) / p
        if not p.exists():
            return StepResult(
                success=False,
                error=f"Library path does not exist: {path}",
            )
        if not p.is_dir():
            return StepResult(
                success=False,
                error=f"SAS library path must be a directory: {path}",
            )
        backend = SasBackend(p, libref=libref, format='xpt')
        session.storage.register(libref, backend)
        return StepResult(
            success=True,
            notes=[f"Library {libref} assigned to {path} (SAS backend: .xpt/.sas7bdat)"],
        )

    if path and engine == "CSV":
        p = Path(path)
        if p.exists() and not p.is_dir():
            return StepResult(
                success=False,
                error=f"CSV library path must be a directory: {path}",
            )
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        backend = CsvBackend(p, libref=libref)
        session.storage.register(libref, backend)
        return StepResult(
            success=True,
            notes=[f"Library {libref} assigned to {path} (CSV engine)"],
        )

    if engine in ("", "MEMORY"):
        # No path — memory backend
        session.storage.register(libref, MemoryBackend())
        return StepResult(
            success=True,
            notes=[f"Library {libref} assigned (memory engine)"],
        )

    return StepResult(
        success=False,
        error=f"LIBNAME engine '{engine}' is not supported. Supported: SAS, XPORT, CSV, MEMORY",
    )
