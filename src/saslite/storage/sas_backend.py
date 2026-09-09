"""SAS file storage backend.

Uses writable SAS Transport (.xpt) files for persistence, and can read existing
SAS7BDAT (.sas7bdat) files when present.

Note: sas7bdat format is read-only. Write operations use XPT format due to
library limitations (pyreadstat does not support sas7bdat write).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd
import pyreadstat
from pandas._libs.sas import get_subheader_index
from pandas.io.sas import sas_constants as sas_const
from pandas.io.sas.sas7bdat import SAS7BDATReader

from saslite.runtime.dataset import Dataset


_MEMBER_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,31}$")

# XPT format column name length limits
# XPT (SAS Transport) format limits
# See: SAS Transport (XPORT) Version 5 specification
# Column names: up to 32 characters (not 8!)
# Table names: up to 8 characters
_XPT_COLUMN_NAME_MAX_LENGTH = 32
_XPT_TABLE_NAME_MAX_LENGTH = 8


def _member_name(name: str) -> str:
    """Validate and normalize SAS member name (dataset name)."""
    member = str(name).upper()
    if not _MEMBER_NAME_RE.fullmatch(member):
        raise ValueError(f"Invalid SAS member name: {name}")
    return member


def _truncate_column_name(name: str, max_length: int = _XPT_COLUMN_NAME_MAX_LENGTH) -> str:
    """Truncate column name to fit XPT format limits.

    Args:
        name: Original column name
        max_length: Maximum allowed length (default: 8 for XPT)

    Returns:
        Truncated column name in uppercase
    """
    normalized = str(name).upper()
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length]


def _prepare_dataframe_for_xpt(df: pd.DataFrame, warn_truncation: bool = True) -> pd.DataFrame:
    """Prepare DataFrame for XPT format export.

    Args:
        df: Input DataFrame
        warn_truncation: Whether to warn about truncated column names

    Returns:
        DataFrame with XPT-compatible column names
    """
    result = df.copy()

    # Track truncations for warning
    truncations = []
    new_columns = []

    for col in result.columns:
        new_col = _truncate_column_name(col)
        if warn_truncation and str(col).upper() != new_col:
            truncations.append((str(col), new_col))
        new_columns.append(new_col)

    # Check for duplicate column names after truncation
    if len(new_columns) != len(set(new_columns)):
        duplicates = [name for name in new_columns if new_columns.count(name) > 1]
        raise ValueError(
            f"Column name truncation resulted in duplicates: {set(duplicates)}. "
            f"Original columns: {list(df.columns)}"
        )

    result.columns = new_columns

    # Emit warning if truncations occurred
    if warn_truncation and truncations:
        trunc_msg = ", ".join(f"'{orig}' -> '{new}'" for orig, new in truncations[:3])
        if len(truncations) > 3:
            trunc_msg += f" (and {len(truncations) - 3} more)"
        warnings.warn(
            f"Column names truncated to {_XPT_COLUMN_NAME_MAX_LENGTH} characters for XPT format: {trunc_msg}",
            UserWarning,
            stacklevel=3
        )

    return result


class _RelaxedSAS7BDATReader(SAS7BDATReader):
    @staticmethod
    def _is_data_subheader(compression: bytes, subheader_compression: int, subheader_type: int) -> bool:
        return bool(
            compression
            and subheader_compression in (sas_const.compressed_subheader_id, 0)
            and subheader_type == sas_const.compressed_subheader_type
        )

    @staticmethod
    def _supports_metadata_page(page_type: int) -> bool:
        return page_type in {
            *sas_const.page_meta_types,
            sas_const.page_amd_type,
            sas_const.page_mix_type,
            sas_const.page_comp_type,
        }

    @staticmethod
    def _supports_data_page(page_type: int) -> bool:
        return page_type in {
            sas_const.page_data_type,
            sas_const.page_mix_type,
            sas_const.page_comp_type,
        }

    def _read_page_header(self) -> None:
        super()._read_page_header()
        self._raw_page_type = self._current_page_type
        if self._current_page_type == sas_const.page_comp_type:
            self._current_page_type = sas_const.page_data_type

    def _process_page_meta(self) -> bool:
        self._read_page_header()
        if self._supports_metadata_page(self._raw_page_type):
            self._process_page_metadata()
        return bool(
            self._supports_data_page(self._raw_page_type)
            or self._current_page_data_subheader_pointers
        )

    def _read_next_page(self):
        self._current_page_data_subheader_pointers = []
        self._cached_page = self._path_or_buf.read(self._page_length)
        if len(self._cached_page) <= 0:
            return True
        if len(self._cached_page) != self._page_length:
            self.close()
            msg = (
                "failed to read complete page from file (read "
                f"{len(self._cached_page):d} of {self._page_length:d} bytes)"
            )
            raise ValueError(msg)

        self._read_page_header()
        if self._supports_metadata_page(self._raw_page_type):
            self._process_page_metadata()

        if not (
            self._supports_data_page(self._raw_page_type)
            or self._raw_page_type in sas_const.page_meta_types
        ):
            return self._read_next_page()

        return False

    def _process_page_metadata(self) -> None:
        bit_offset = self._page_bit_offset

        for i in range(self._current_page_subheaders_count):
            offset = sas_const.subheader_pointers_offset + bit_offset
            total_offset = offset + self._subheader_pointer_length * i

            subheader_offset = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length

            subheader_length = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length

            subheader_compression = self._read_uint(total_offset, 1)
            total_offset += 1

            subheader_type = self._read_uint(total_offset, 1)

            if (
                subheader_length == 0
                or subheader_compression == sas_const.truncated_subheader_id
            ):
                continue

            subheader_signature = self._read_bytes(subheader_offset, self._int_length)

            try:
                subheader_index = get_subheader_index(subheader_signature)
            except ValueError:
                if self._is_data_subheader(
                    self.compression,
                    subheader_compression,
                    subheader_type,
                ):
                    self._current_page_data_subheader_pointers.append(
                        (subheader_offset, subheader_length)
                    )
                continue

            subheader_processor = self._subheader_processors[subheader_index]
            if subheader_processor is None:
                if self._is_data_subheader(
                    self.compression,
                    subheader_compression,
                    subheader_type,
                ):
                    self._current_page_data_subheader_pointers.append(
                        (subheader_offset, subheader_length)
                    )
                continue

            subheader_processor(subheader_offset, subheader_length)


def _decode_sas_string(value: object, encoding: str) -> object:
    if not isinstance(value, bytes):
        return value

    cleaned = value.replace(b"\x00", b"").rstrip(b" \xa0")
    if not cleaned:
        return ""

    encodings = [encoding, "utf-8", "latin1"]
    tried: set[str] = set()
    for candidate in encodings:
        if not candidate or candidate in tried:
            continue
        tried.add(candidate)
        try:
            return cleaned.decode(candidate)
        except UnicodeDecodeError:
            continue

    return cleaned.decode(encoding or "utf-8", errors="replace")


def _decode_sas_dataframe(df: pd.DataFrame, encoding: str) -> pd.DataFrame:
    decoded = df.copy()
    for col in decoded.columns:
        if decoded[col].dtype == object:
            decoded[col] = decoded[col].map(lambda value: _decode_sas_string(value, encoding))
    return decoded


def _read_sas7bdat_relaxed(path: Path) -> pd.DataFrame:
    reader = _RelaxedSAS7BDATReader(path, encoding="utf-8", convert_text=False, convert_dates=False)
    try:
        df = reader.read()
        return _decode_sas_dataframe(df, reader.encoding or reader.default_encoding)
    finally:
        reader.close()


def _read_sas7bdat(path: Path) -> pd.DataFrame:
    # ReadStat honors the file's encoding. Keep SAS date/time values numeric
    # (days/seconds since 1960), as expected by DATA-step arithmetic.
    try:
        df, meta = pyreadstat.read_sas7bdat(str(path), disable_datetime_conversion=True)
        df.attrs['sas_labels'] = meta.column_names_to_labels
        df.attrs['sas_formats'] = meta.original_variable_types
        return df
    except pyreadstat.PyreadstatError:
        pass
    try:
        with SAS7BDATReader(path, encoding="utf-8", convert_dates=False) as reader:
            return reader.read()
    except ValueError as exc:
        if "Unknown subheader signature" not in str(exc):
            raise
        return _read_sas7bdat_relaxed(path)


class SasBackend:
    """SAS file-based dataset storage.

    Supports reading both XPT (Transport) and sas7bdat formats.
    Write operations always use XPT format due to library limitations.

    Args:
        base_dir: Base directory for dataset storage
        libref: Library reference name (default: "DISK")
        format: Desired output format - 'sas7bdat' (default) or 'xpt'
        warn_truncation: Whether to warn when column names are truncated for XPT format (default: True)

    Native SAS7BDAT input is supported. Native writes are not supported by the
    underlying reader. Use format='xpt' for genuine SAS Transport output;
    the backend never disguises newly written XPT as native SAS7BDAT.

    Example:
        >>> # Read native SAS datasets
        >>> backend = SasBackend('./work')
        >>>
        >>> # Explicitly specify XPT format
        >>> backend = SasBackend('./work', format='xpt')
    """

    def __init__(
        self,
        base_dir: str | Path,
        libref: str = "DISK",
        format: Literal["sas7bdat", "xpt"] = "sas7bdat",
        warn_truncation: bool = True,
    ) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._libref = libref
        self._format = format
        self._warn_truncation = warn_truncation
        self.engine = "SAS"
        self.path = self._base

        # Validate format
        if format not in ("sas7bdat", "xpt"):
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: 'sas7bdat', 'xpt'"
            )

    def _find_existing_path(self, name: str, suffix: str) -> Path:
        """Find a dataset file using SAS-style case-insensitive member names."""
        member = _member_name(name)
        exact = self._base / f"{member}{suffix}"
        if exact.exists():
            return exact

        if not self._base.exists():
            return exact

        for candidate in self._base.iterdir():
            if (
                candidate.is_file()
                and candidate.suffix.lower() == suffix.lower()
                and candidate.stem.upper() == member
            ):
                return candidate

        return exact

    def _xpt_path(self, name: str) -> Path:
        return self._find_existing_path(name, ".xpt")

    def _sas7bdat_path(self, name: str) -> Path:
        return self._find_existing_path(name, ".sas7bdat")

    def read(self, name: str) -> Dataset | None:
        member = _member_name(name)

        # A real native file takes precedence over a transport file of the
        # same name. Recognize legacy mislabeled XPT by its header explicitly.
        native_path = self._sas7bdat_path(member)
        if native_path.exists():
            with native_path.open('rb') as stream:
                transport = stream.read(20).startswith(b'HEADER RECORD')
            if transport:
                df, _ = pyreadstat.read_xport(str(native_path))
            else:
                df = _read_sas7bdat(native_path)
            dataset = Dataset.from_dataframe(df, name=member, libref=self._libref)
            for column in dataset.metadata.variables.values():
                column.label = df.attrs.get('sas_labels', {}).get(column.name)
                column.format = df.attrs.get('sas_formats', {}).get(column.name)
            return dataset

        # Try XPT path first
        xpt_path = self._xpt_path(member)
        if xpt_path.exists():
            df, _ = pyreadstat.read_xport(str(xpt_path))
            return Dataset.from_dataframe(df, name=member, libref=self._libref)

        return None

    def write(self, name: str, dataset: Dataset) -> None:
        """Write dataset to disk in the configured format.

        Args:
            name: Dataset member name
            dataset: Dataset to write

        Native writes are rejected explicitly; XPT writes use .xpt names.
        """
        member = _member_name(name)

        if self._format == 'sas7bdat':
            raise NotImplementedError('Native SAS7BDAT writing is not supported; use WORK for temporary data or XPORT for persistent output')
        if self._sas7bdat_path(member).exists():
            raise FileExistsError(f'Cannot replace native member {member} with XPT; use WORK or a different output member name')
        path = self._xpt_path(member)

        # Always use XPT format for writing (due to library limitations)
        df = _prepare_dataframe_for_xpt(dataset.data, warn_truncation=self._warn_truncation)
        table_name = member[:_XPT_TABLE_NAME_MAX_LENGTH]
        pyreadstat.write_xport(df, str(path), table_name=table_name)

    def exists(self, name: str) -> bool:
        member = _member_name(name)
        return self._xpt_path(member).exists() or self._sas7bdat_path(member).exists()

    def delete(self, name: str) -> bool:
        member = _member_name(name)
        deleted = False
        for path in (self._xpt_path(member), self._sas7bdat_path(member)):
            if path.exists():
                path.unlink()
                deleted = True
        return deleted

    def list_datasets(self) -> list[str]:
        if not self._base.exists():
            return []
        names: set[str] = set()
        for path in self._base.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in {".xpt", ".sas7bdat"}
                and _MEMBER_NAME_RE.fullmatch(path.stem.upper())
            ):
                names.add(path.stem.upper())
        return sorted(names)
