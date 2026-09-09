"""Macro preprocessor: expands %LET, &var, %MACRO/%MEND, and handles comments."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime as _datetime
from typing import Any


@dataclass
class MacroDef:
    """Definition of a SAS macro."""
    name: str
    params: list[str] = field(default_factory=list)
    body: str = ""
    defaults: dict[str, str] = field(default_factory=dict)


class MacroExpander:
    """Macro expander for %LET, &var, %MACRO/%MEND, and macro invocation."""

    def __init__(self, session: Any | None = None) -> None:
        self._global_vars: dict[str, str] = {}
        self._local_vars: dict[str, str] = {}
        self._macros: dict[str, MacroDef] = {}
        self._put_output: list[str] = []
        self._session = session
        self._init_automatic_vars()

    def _init_automatic_vars(self) -> None:
        """Populate SAS automatic macro variables (computed once per session)."""
        now = _datetime.now()
        self._global_vars["SYSDATE"] = now.strftime("%d%b%y").upper()
        self._global_vars["SYSDATE9"] = now.strftime("%d%b%Y").upper()
        self._global_vars["SYSTIME"] = now.strftime("%H:%M")
        self._global_vars["SYSDAY"] = now.strftime("%A")
        self._global_vars["SYSVER"] = "9.4"
        self._global_vars["SYSERR"] = "0"
        if sys.platform.startswith("win"):
            self._global_vars["SYSSCP"] = "WIN"
        elif sys.platform == "darwin":
            self._global_vars["SYSSCP"] = "MAC"
        else:
            self._global_vars["SYSSCP"] = "LIN X64"

    def set_var(self, name: str, value: str) -> None:
        self._local_vars[name.upper()] = value

    def get_var(self, name: str) -> str | None:
        key = name.upper()
        if key in self._local_vars:
            return self._local_vars[key]
        if key in self._global_vars:
            return self._global_vars[key]
        return None

    def _log_symbolgen(self, var_name: str, value: str) -> None:
        """Log SYMBOLGEN output if enabled."""
        if self._session and self._session.get_option("SYMBOLGEN", False):
            msg = f"SYMBOLGEN:  Macro variable {var_name.upper()} resolves to {value}"
            self._session.add_debug_output(msg)

    def _log_mprint(self, text: str) -> None:
        """Log MPRINT output if enabled."""
        if self._session and self._session.get_option("MPRINT", False):
            msg = f"MPRINT:  {text.strip()}"
            self._session.add_debug_output(msg)

    def _log_mlogic(self, message: str) -> None:
        """Log MLOGIC output if enabled."""
        if self._session and self._session.get_option("MLOGIC", False):
            msg = f"MLOGIC:  {message}"
            self._session.add_debug_output(msg)

    @property
    def put_output(self) -> list[str]:
        """Collected %PUT output lines."""
        return self._put_output

    def expand(self, source: str) -> str:
        """Expand macro variables and process %LET / %MACRO statements.

        This is the main entry point. It:
        1. Removes block comments /* ... */
        2. Converts SAS format/informat literals (e.g. e8601da.) to quoted strings
        3. Processes %MACRO/%MEND definitions
        4. Processes %LET statements (must be before conditionals)
        5. Processes %DO %TO %BY iterative loops
        6. Substitutes &var references
        7. Processes %IF/%THEN/%ELSE conditionals
        8. Expands macro invocations (%name)
        9. Final &var substitution pass
        """
        # Step 1: Remove block comments
        source = self._remove_comments(source)

        # Step 1.5: Convert format/informat literals to quoted strings
        source = self._quote_format_literals(source)

        # Step 2: Process %MACRO/%MEND definitions
        source = self._process_macro_definitions(source)

        # Step 2.5: Process macro character functions and %SYSFUNC
        source = self._process_macro_functions(source)

        # Step 3: Process %DO %TO %BY iterative loops
        source = self._process_do_loops(source)

        # Step 4: Process %EVAL/%SYSEVALF expressions
        source = self._process_eval(source)

        # Step 5: Process %LET statements (after eval so values resolve)
        source = self._process_let_statements(source)

        # Step 5.5: Macro functions may now resolve with %LET values
        source = self._process_macro_functions(source)

        # Step 6: Substitute &var references
        source = self._substitute_vars(source)

        # Step 7: Process %IF/%THEN/%ELSE conditionals
        source = self._process_conditionals(source)

        # Step 8: Process %PUT statements
        source = self._process_put_statements(source)

        # Step 9: Expand macro invocations
        source = self._expand_macro_invocations(source)

        # Step 9.5: The expanded macro bodies may contain %DO/%EVAL/%LET/
        # %IF/%PUT and macro functions — process them now.
        source = self._process_macro_functions(source)
        source = self._process_do_loops(source)
        source = self._process_eval(source)
        source = self._process_let_statements(source)
        source = self._substitute_vars(source)
        source = self._process_conditionals(source)
        source = self._process_put_statements(source)

        # Step 10: Final &var substitution pass
        source = self._substitute_vars(source)

        return source

    def _remove_comments(self, source: str) -> str:
        """Remove /* ... */ block comments, %* ... ; macro comments, and
        inline statement comments (`* ... ;` directly after a semicolon)."""
        while "/*" in source:
            source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
        source = re.sub(r"%\*[^;]*;", "", source)
        # `;  * comment ;` — a star where a new statement would begin
        source = re.sub(r"(?<=;)[ \t]*\*[^;\n]*;", "", source)
        return source

    def _quote_format_literals(self, source: str) -> str:
        """Convert unquoted SAS format/informat literals to quoted strings.

        SAS format/informat names always end with a dot (e.g. e8601da., date9., best32.).
        They appear as the second argument to INPUT() and PUT() functions.
        Convert:  input(x, e8601da.)  ->  input(x, 'e8601da.')
                  put(x, e8601da.)    ->  put(x, 'e8601da.')
        """
        # Use a character-level scanner to find INPUT/PUT calls and quote format args
        result = []
        i = 0
        upper = source.upper()
        n = len(source)
        while i < n:
            # Look for INPUT( or PUT(
            if (upper[i:i+5] == 'INPUT' or upper[i:i+3] == 'PUT') and i > 0 and (source[i-1].isalnum() or source[i-1] == '_'):
                result.append(source[i])
                i += 1
                continue
            if upper[i:i+5] == 'INPUT' and i + 5 < n and not source[i+5].isalnum() and source[i+5] != '_':
                func_end = i + 5
                result.append(source[i:func_end])
                i = func_end
            elif upper[i:i+3] == 'PUT' and i + 3 < n and not source[i+3].isalnum() and source[i+3] != '_':
                func_end = i + 3
                result.append(source[i:func_end])
                i = func_end
            else:
                result.append(source[i])
                i += 1
                continue
            # Skip whitespace to find (
            while i < n and source[i] in ' \t\n':
                result.append(source[i])
                i += 1
            if i >= n or source[i] != '(':
                continue
            # Found opening paren — find matching closing paren
            result.append(source[i])
            i += 1
            depth = 1
            inner_start = i
            while i < n and depth > 0:
                if source[i] == '(':
                    depth += 1
                elif source[i] == ')':
                    depth -= 1
                i += 1
            inner = source[inner_start:i-1]  # content between parens
            closing = source[i-1]  # )
            # Split inner by comma at depth 0
            parts = self._split_func_args(inner)
            if len(parts) >= 2:
                last_arg = parts[-1].strip()
                # Check if it's an unquoted format literal
                if re.match(r'^[A-Za-z_]\w*\.(\d+\.?(\d+)?)?$', last_arg) or \
                   re.match(r'^\d+\.\d*$', last_arg):
                    parts[-1] = f"'{last_arg}'"
                    result.append(','.join(parts))
                    result.append(closing)
                    continue
            # No quoting needed — output as-is
            result.append(inner)
            result.append(closing)
        return ''.join(result)

    def _split_func_args(self, inner: str) -> list[str]:
        """Split function arguments by comma, respecting parentheses depth."""
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in inner:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts

    def _process_macro_definitions(self, source: str) -> str:
        """Find and extract %MACRO ... %MEND blocks."""
        # Match %MACRO name[(params)] ... %MEND [name];
        pattern = r"%\s*MACRO\s+(\w+)(?:\s*\(([^)]*)\))?\s*;(.*?)%\s*MEND(?:\s+\w+)?\s*;"
        result = source
        for match in re.finditer(pattern, source, flags=re.IGNORECASE | re.DOTALL):
            name = match.group(1).upper()
            params_str = match.group(2) or ""
            body = match.group(3)
            params: list[str] = []
            defaults: dict[str, str] = {}
            for p in params_str.split(","):
                p = p.strip()
                if not p:
                    continue
                if "=" in p:
                    pname, pdefault = p.split("=", 1)
                    pname = pname.strip().upper()
                    params.append(pname)
                    defaults[pname] = pdefault.strip()
                else:
                    params.append(p.upper())
            self._macros[name] = MacroDef(name=name, params=params, body=body,
                                          defaults=defaults)
            result = result.replace(match.group(0), "")
        return result

    def _process_conditionals(self, source: str) -> str:
        """Process %IF ... %THEN ... %ELSE ... %DO ... %END; conditionals."""
        max_iterations = 20
        for _ in range(max_iterations):
            new_source = self._process_conditionals_once(source)
            if new_source == source:
                break
            source = new_source
        return source

    def _process_conditionals_once(self, source: str) -> str:
        """Single pass of %IF/%THEN/%ELSE processing."""
        # Pattern for %IF ... %THEN %DO ... %END [%ELSE %DO ... %END]
        # First handle %IF ... %THEN %DO ... %END [%ELSE %DO ... %END]
        do_pattern = (
            r"%\s*IF\s+(.*?)\s+%\s*THEN\s+%\s*DO\s*;"
            r"(.*?)%\s*END\s*;"
            r"(?:\s*%\s*ELSE\s+%\s*DO\s*;(.*?)%\s*END\s*;)?"
        )
        source = re.sub(
            do_pattern,
            lambda m: self._eval_conditional(m),
            source,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Pattern for %IF ... %THEN single_statement;
        # (no %DO/%END, just a single statement)
        single_pattern = (
            r"%\s*IF\s+(.*?)\s+%\s*THEN\s+([^;]+;)"
            r"(?:\s*%\s*ELSE\s+([^;]+;))?"
        )
        source = re.sub(
            single_pattern,
            lambda m: self._eval_conditional_single(m),
            source,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return source

    def _eval_conditional(self, match: re.Match) -> str:
        """Evaluate a %IF/%THEN/%DO conditional and return the chosen branch."""
        condition = match.group(1).strip()
        then_body = match.group(2)
        else_body = match.group(3) if match.group(3) else ""
        if self._eval_macro_condition(condition):
            return then_body
        return else_body

    def _eval_conditional_single(self, match: re.Match) -> str:
        """Evaluate a %IF/%THEN single-statement conditional."""
        condition = match.group(1).strip()
        then_body = match.group(2)
        else_body = match.group(3) if match.group(3) else ""
        if self._eval_macro_condition(condition):
            return then_body
        return else_body

    def _eval_macro_condition(self, condition: str) -> bool:
        """Evaluate a macro-level condition like 'X = 1' or 'X NE Y'."""
        # Substitute any remaining &vars
        condition = self._substitute_vars(condition)

        # Handle comparison operators
        for op in ("<>", " NE ", " GE ", " LE ", " GT ", " LT ", ">=", "<=", "=", ">", "<"):
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    left = parts[0].strip().strip("'\"")
                    right = parts[1].strip().strip("'\"")
                    try:
                        left_n = float(left)
                        right_n = float(right)
                        left, right = left_n, right_n
                    except ValueError:
                        pass
                    if op.strip() in ("=", ""):
                        return left == right
                    if op.strip() in ("NE", "<>"):
                        return left != right
                    if op.strip() == ">":
                        return left > right
                    if op.strip() == ">=" or op.strip() == "GE":
                        return left >= right
                    if op.strip() == "<":
                        return left < right
                    if op.strip() == "<=" or op.strip() == "LE":
                        return left <= right
                    if op.strip() == "GT":
                        return left > right
                    if op.strip() == "LT":
                        return left < right
        # If no operator, check if non-empty / non-zero
        val = condition.strip().strip("'\"")
        return val not in ("", "0", ".", "FALSE")

    def _expand_macro_invocations(self, source: str) -> str:
        """Expand %macro_name and %macro_name(args) invocations."""
        max_iterations = 20  # Prevent infinite loops
        for _ in range(max_iterations):
            expanded = self._expand_once(source)
            if expanded == source:
                break
            source = expanded
        return source

    @staticmethod
    def _apply_return_goto(body: str) -> str:
        """Apply %RETURN and %GOTO/%label semantics to an expanded macro body.

        %RETURN; truncates the rest of the body. %GOTO name; skips forward
        to the matching %name: label (backward jumps are ignored to avoid
        infinite loops). Remaining label markers are removed.
        """
        # %RETURN — truncate
        m = re.search(r"%\s*RETURN\s*;", body, flags=re.IGNORECASE)
        if m:
            body = body[:m.start()]

        # %GOTO label — forward jump only
        for _ in range(20):
            g = re.search(r"%\s*GOTO\s+(\w+)\s*;", body, flags=re.IGNORECASE)
            if g is None:
                break
            label = g.group(1)
            lbl = re.compile(rf"%\s*{re.escape(label)}\s*:", flags=re.IGNORECASE)
            target = lbl.search(body, g.end())
            if target:
                body = body[:g.start()] + body[target.end():]
            else:
                # Label not found ahead — drop the %GOTO and continue
                body = body[:g.start()] + body[g.end():]

        # Strip leftover label markers
        body = re.sub(r"%\s*\w+\s*:", "", body)
        return body

    def _expand_once(self, source: str) -> str:
        """Single pass of macro expansion."""
        # Match %name(args) or %name (but not %LET, %MACRO, %MEND, %IF, %DO, %THEN, %ELSE, %END)
        # %name with arguments: %name(arg1, arg2) — trailing ; optional
        pattern_with_args = r"%(\w+)\s*\(([^)]*)\)\s*;?"
        # %name without arguments: %name;
        pattern_no_args = r"%(\w+)\s*;"

        skip_keywords = {"LET", "MACRO", "MEND", "IF", "THEN", "ELSE", "DO", "END",
                         "PUT", "INCLUDE", "GOTO", "RETURN", "EVAL", "SYSEVALF",
                         "UPCASE", "LOWCASE", "SCAN", "SUBSTR", "LENGTH", "INDEX",
                         "SYSFUNC", "TO", "BY"}

        # First expand %name(args);
        def replacer_with_args(match: re.Match) -> str:
            macro_name = match.group(1).upper()
            if macro_name in skip_keywords:
                return match.group(0)
            if macro_name not in self._macros:
                return match.group(0)
            macro = self._macros[macro_name]
            args_str = match.group(2)
            args = [a.strip() for a in self._split_args_depth0(args_str)]
            # Build local variable scope: defaults first, then positional
            # and keyword (name=value) arguments
            local_vars: dict[str, str] = dict(macro.defaults)
            pos_idx = 0
            for arg in args:
                if not arg:
                    pos_idx += 1
                    continue
                kw = re.match(r"^(\w+)\s*=\s*(.*)$", arg, flags=re.DOTALL)
                if kw and kw.group(1).upper() in macro.params:
                    local_vars[kw.group(1).upper()] = kw.group(2).strip().strip("'\"")
                else:
                    if pos_idx < len(macro.params):
                        local_vars[macro.params[pos_idx]] = arg.strip("'\"")
                    pos_idx += 1
            # Expand body with local vars
            body = macro.body
            for var_name, var_val in local_vars.items():
                body = re.sub(rf"&{re.escape(var_name)}\b\.?", var_val, body, flags=re.IGNORECASE)
            # Also substitute global/local vars
            body = self._substitute_vars(body)
            return self._apply_return_goto(body)

        source = re.sub(pattern_with_args, replacer_with_args, source, flags=re.IGNORECASE)

        # Then expand %name; (no args)
        def replacer_no_args(match: re.Match) -> str:
            macro_name = match.group(1).upper()
            if macro_name in skip_keywords:
                return match.group(0)
            if macro_name not in self._macros:
                return match.group(0)
            macro = self._macros[macro_name]
            body = self._substitute_vars(macro.body)
            return self._apply_return_goto(body)

        source = re.sub(pattern_no_args, replacer_no_args, source, flags=re.IGNORECASE)
        return source

    def _process_let_statements(self, source: str) -> str:
        """Find and process %LET statements, return source without them."""
        def replacer(match: re.Match) -> str:
            var_name = match.group(1).upper()
            value = match.group(2).strip()
            # Remove surrounding quotes if present
            if (value.startswith("'") and value.endswith("'")) or (
                value.startswith('"') and value.endswith('"')
            ):
                value = value[1:-1]
            self._local_vars[var_name] = value
            self._log_symbolgen(var_name, value)
            return ""  # Remove the %LET statement

        source = re.sub(
            r"%\s*LET\s+(\w+)\s*=\s*(.*?);",
            replacer,
            source,
            flags=re.IGNORECASE,
        )
        return source

    def _substitute_vars(self, source: str) -> str:
        """Substitute &var references with their values.

        SAS semantics: a trailing dot after a macro variable reference is a
        delimiter and is consumed (`&lib..member` → `value.member`).
        """
        # Handle &&var (indirect reference) — resolve one level
        def _indirect(m: re.Match) -> str:
            val = self.get_var(m.group(1))
            if val is None:
                return m.group(0)
            self._log_symbolgen(m.group(1), val)
            return val + (m.group(2) or "")

        source = re.sub(r"&&(\w+)(\.)?", _indirect, source)

        # Handle &var with optional trailing-dot delimiter
        def _direct(m: re.Match) -> str:
            val = self.get_var(m.group(1))
            if val is None:
                return m.group(0)  # keep unresolved reference (incl. dot)
            self._log_symbolgen(m.group(1), val)
            return val

        source = re.sub(r"&(\w+)\.?", _direct, source)
        return source

    def _process_put_statements(self, source: str) -> str:
        """Process %PUT statements — write text to output log."""
        def replacer(match: re.Match) -> str:
            content = match.group(1).strip()
            # Handle special %PUT keywords
            if content.upper() == "_ALL_":
                # Output all macro variables
                lines = []
                for name, value in sorted(self._global_vars.items()):
                    lines.append(f"{name}={value}")
                for name, value in sorted(self._local_vars.items()):
                    lines.append(f"{name}={value}")
                output = "\n".join(lines)
            elif content.upper() == "_AUTOMATIC_":
                # Output automatic macro variables
                lines = []
                auto_vars = ["SYSDATE", "SYSDATE9", "SYSTIME", "SYSDAY", "SYSVER", "SYSERR", "SYSSCP"]
                for name in auto_vars:
                    if name in self._global_vars:
                        lines.append(f"{name}={self._global_vars[name]}")
                output = "\n".join(lines)
            elif content.upper() == "_USER_":
                # Output user-defined macro variables
                lines = []
                for name, value in sorted(self._local_vars.items()):
                    lines.append(f"{name}={value}")
                output = "\n".join(lines)
            else:
                # Substitute variables in the content
                output = self._substitute_vars(content)
                # Evaluate any %EVAL expressions
                output = self._process_eval(output)

            self._put_output.append(output)
            if self._session:
                self._session.add_debug_output(f"PUT: {output}")
            return ""

        source = re.sub(
            r"%\s*PUT\s+(.*?);",
            replacer,
            source,
            flags=re.IGNORECASE,
        )
        return source

    def _process_do_loops(self, source: str) -> str:
        """Process %DO %var = %eval(start) %TO %eval(end) %BY step; ... %END; loops."""
        max_iterations = 50
        for _ in range(max_iterations):
            new_source = self._process_do_loops_once(source)
            if new_source == source:
                break
            source = new_source
        return source

    def _process_do_loops_once(self, source: str) -> str:
        """Single pass of %DO loop expansion."""
        # Pattern: %DO var = start %TO end [%BY step]; body %END;
        pattern = (
            r"%\s*DO\s+(\w+)\s*=\s*(.*?)\s+%\s*TO\s+(.*?)"
            r"(?:\s+%\s*BY\s+(.*?))?\s*;(.*?)%\s*END\s*;"
        )

        def expand_loop(match: re.Match) -> str:
            var_name = match.group(1).upper()
            start_str = self._substitute_vars(match.group(2).strip())
            end_str = self._substitute_vars(match.group(3).strip())
            by_str = match.group(4)
            body = match.group(5)

            # Process %EVAL in start/end
            start_str = self._process_eval(start_str)
            end_str = self._process_eval(end_str)

            try:
                start = int(float(start_str))
                end = int(float(end_str))
            except (ValueError, TypeError):
                return match.group(0)  # Can't expand, leave as-is

            by = 1
            if by_str:
                by_str = self._substitute_vars(by_str.strip())
                by_str = self._process_eval(by_str)
                try:
                    by = int(float(by_str))
                except (ValueError, TypeError):
                    by = 1
            if by == 0:
                by = 1

            # Expand the loop
            result_parts = []
            i = start
            while (i <= end if by > 0 else i >= end):
                # Substitute &var with current value in body
                expanded_body = re.sub(
                    rf"&{re.escape(var_name)}\b",
                    str(i),
                    body,
                    flags=re.IGNORECASE,
                )
                # Also process nested constructs
                expanded_body = self._substitute_vars(expanded_body)
                expanded_body = self._process_eval(expanded_body)
                result_parts.append(expanded_body)
                i += by

            return "".join(result_parts)

        source = re.sub(
            pattern,
            expand_loop,
            source,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return source

    # ── Macro character functions & %SYSFUNC ──────────

    _MACRO_FUNC_NAMES = ("UPCASE", "LOWCASE", "SCAN", "SUBSTR", "LENGTH",
                         "INDEX", "SYSFUNC")

    def _process_macro_functions(self, source: str) -> str:
        """Expand %UPCASE/%LOWCASE/%SCAN/%SUBSTR/%LENGTH/%INDEX/%SYSFUNC.

        Uses a left-to-right scanner with balanced-paren matching. Calls whose
        arguments still contain a nested macro-function call are skipped in
        the current pass (the inner call is expanded first); the pass repeats
        until the source is stable.
        """
        names = "|".join(self._MACRO_FUNC_NAMES)
        head_re = re.compile(rf"%\s*({names})\s*\(", flags=re.IGNORECASE)

        for _ in range(30):
            new_source = self._expand_macro_functions_once(source, head_re)
            if new_source == source:
                break
            source = new_source
        return source

    def _expand_macro_functions_once(self, source: str, head_re: re.Pattern) -> str:
        result: list[str] = []
        pos = 0
        n = len(source)
        while pos < n:
            m = head_re.search(source, pos)
            if m is None:
                result.append(source[pos:])
                break
            func = m.group(1).upper()
            # Find the balanced closing paren
            depth = 1
            i = m.end()
            while i < n and depth > 0:
                if source[i] == "(":
                    depth += 1
                elif source[i] == ")":
                    depth -= 1
                i += 1
            if depth != 0:
                # Unbalanced — leave the rest untouched
                result.append(source[pos:])
                break
            raw_args = source[m.end():i - 1]
            if head_re.search(raw_args):
                # Nested macro function inside — expand inner first.
                # Emit up to (and including) the function head, then continue
                # scanning inside the args.
                result.append(source[pos:m.end()])
                pos = m.end()
                continue
            result.append(source[pos:m.start()])
            expanded = self._eval_macro_function(func, raw_args)
            if expanded is None:
                # Arguments not yet resolvable (unresolved &var) — keep as-is
                result.append(source[m.start():i])
            else:
                result.append(expanded)
            pos = i
        return "".join(result)

    def _eval_macro_function(self, func: str, raw_args: str) -> str | None:
        """Evaluate one macro character function call.

        Returns None when the arguments still contain unresolved &var
        references, so expansion can be retried after %LET processing.
        """
        raw_args = self._substitute_vars(raw_args)
        if "&" in raw_args:
            return None

        if func == "SYSFUNC":
            return self._eval_sysfunc(raw_args)

        args = [a.strip() for a in raw_args.split(",")]
        text = args[0] if args else ""

        if func == "UPCASE":
            return text.upper()
        if func == "LOWCASE":
            return text.lower()
        if func == "LENGTH":
            return str(len(text))
        if func == "INDEX":
            target = args[1] if len(args) > 1 else ""
            return str(text.find(target) + 1)
        if func == "SCAN":
            n = 1
            if len(args) > 1:
                try:
                    n = int(float(args[1]))
                except ValueError:
                    n = 1
            delims = args[2] if len(args) > 2 and args[2] else " \t\n\r.-/,;:!?()[]{}"
            words = [w for w in re.split(f"[{re.escape(delims)}]+", text) if w]
            if n < 0:
                n = len(words) + n + 1
            if 1 <= n <= len(words):
                return words[n - 1]
            return ""
        if func == "SUBSTR":
            try:
                start = int(float(args[1])) if len(args) > 1 else 1
            except ValueError:
                start = 1
            length = None
            if len(args) > 2 and args[2]:
                try:
                    length = int(float(args[2]))
                except ValueError:
                    length = None
            start_idx = max(start - 1, 0)
            if length is None:
                return text[start_idx:]
            return text[start_idx:start_idx + length]
        return ""

    _sysfunc_registry = None

    def _eval_sysfunc(self, raw_args: str) -> str:
        """Evaluate %SYSFUNC(func(args) [, format])."""
        m = re.match(r"\s*(\w+)\s*\((.*)\)\s*(?:,\s*[\w.$]+\s*)?$", raw_args, flags=re.DOTALL)
        if m is None:
            # Function with no parens: %SYSFUNC(today())  handled above;
            # %SYSFUNC(today) style:
            m2 = re.match(r"\s*(\w+)\s*$", raw_args)
            if m2 is None:
                return ""
            fn_name, inner = m2.group(1), ""
        else:
            fn_name, inner = m.group(1), m.group(2)

        if MacroExpander._sysfunc_registry is None:
            from saslite.functions import build_default_registry
            MacroExpander._sysfunc_registry = build_default_registry()
        reg = MacroExpander._sysfunc_registry

        args: list[Any] = []
        if inner.strip():
            for part in self._split_args_depth0(inner):
                part = part.strip()
                if (len(part) >= 2 and part[0] in ("'", '"') and part[-1] == part[0]):
                    args.append(part[1:-1])
                    continue
                try:
                    num = float(part)
                    args.append(int(num) if num.is_integer() else num)
                except ValueError:
                    args.append(part)

        try:
            result = reg.call(fn_name, args)
        except NameError:
            raise
        except Exception:
            return ""
        if result is None:
            return ""
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    @staticmethod
    def _split_args_depth0(inner: str) -> list[str]:
        """Split a comma-separated arg string at parenthesis depth 0."""
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in inner:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    def _process_eval(self, source: str) -> str:
        """Process %EVAL() and %SYSEVALF() expressions."""
        def eval_replacer(match: re.Match) -> str:
            expr = match.group(1).strip()
            expr = self._substitute_vars(expr)
            return str(self._eval_macro_expr(expr))

        # %SYSEVALF(expr) — floating-point arithmetic
        def sysevalf_replacer(match: re.Match) -> str:
            expr = match.group(1).strip()
            expr = self._substitute_vars(expr)
            return str(self._eval_macro_expr(expr, floating=True))

        source = re.sub(
            r"%\s*EVAL\s*\(([^)]+)\)",
            eval_replacer,
            source,
            flags=re.IGNORECASE,
        )
        source = re.sub(
            r"%\s*SYSEVALF\s*\(([^)]+)\)",
            sysevalf_replacer,
            source,
            flags=re.IGNORECASE,
        )
        return source

    @staticmethod
    def _eval_macro_expr(expr: str, floating: bool = False) -> int | float:
        """Evaluate a macro-level arithmetic/comparison expression."""
        expr = expr.strip()
        # Try direct number
        try:
            return float(expr) if floating else int(float(expr))
        except (ValueError, TypeError):
            pass

        # Comparison operators (lowest precedence) — return 1/0
        comp_ops: list[tuple[str, Any]] = [
            (">=", lambda a, b: a >= b), ("<=", lambda a, b: a <= b),
            ("=", lambda a, b: a == b), (">", lambda a, b: a > b),
            ("<", lambda a, b: a < b),
        ]
        mnemonic_ops: list[tuple[str, Any]] = [
            ("NE", lambda a, b: a != b), ("GE", lambda a, b: a >= b),
            ("LE", lambda a, b: a <= b), ("GT", lambda a, b: a > b),
            ("LT", lambda a, b: a < b), ("EQ", lambda a, b: a == b),
        ]

        def _operand(s: str) -> Any:
            s = s.strip()
            try:
                val = MacroExpander._eval_macro_expr(s, floating=True)
                # _eval_macro_expr returns 0 for unknown text — but for
                # comparisons we want string semantics there. Distinguish:
                # if s parses as expr containing digits/operators, trust it.
                if re.search(r"\d", s) or any(op in s for op in "+-*/()"):
                    return val
            except Exception:
                pass
            return s.strip("'\"")

        # Mnemonic operators need word boundaries
        for op_word, op_fn in mnemonic_ops:
            m = re.search(rf"\b{op_word}\b", expr, flags=re.IGNORECASE)
            if m:
                left = _operand(expr[:m.start()])
                right = _operand(expr[m.end():])
                try:
                    return 1 if op_fn(float(left), float(right)) else 0
                except (TypeError, ValueError):
                    return 1 if op_fn(str(left), str(right)) else 0

        # Symbolic comparison operators at depth 0
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0:
                for op_sym, op_fn in comp_ops:
                    if expr[i:i + len(op_sym)] == op_sym:
                        # Skip ** which is not comparison; = is unambiguous
                        left = _operand(expr[:i])
                        right = _operand(expr[i + len(op_sym):])
                        try:
                            return 1 if op_fn(float(left), float(right)) else 0
                        except (TypeError, ValueError):
                            return 1 if op_fn(str(left), str(right)) else 0

        # Handle common arithmetic operators
        ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
               ("*", lambda a, b: a * b), ("/", lambda a, b: a / b if b != 0 else 0)]

        for op_sym, op_fn in ops:
            # Find the operator (respecting parentheses depth)
            depth = 0
            split_pos = -1
            for i, ch in enumerate(expr):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif depth == 0 and ch == op_sym and i > 0:
                    split_pos = i
                    break  # Take the first occurrence at depth 0
            if split_pos > 0:
                left = MacroExpander._eval_macro_expr(expr[:split_pos], floating)
                right = MacroExpander._eval_macro_expr(expr[split_pos + 1:], floating)
                return op_fn(left, right)

        # Handle parentheses
        if expr.startswith("(") and expr.endswith(")"):
            return MacroExpander._eval_macro_expr(expr[1:-1], floating)

        # Unknown expression — return 0
        return 0
