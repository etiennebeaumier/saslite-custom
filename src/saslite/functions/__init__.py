"""Functions package — register all built-in SAS functions."""

from __future__ import annotations

from saslite.functions.registry import FunctionRegistry


def build_default_registry() -> FunctionRegistry:
    """Build a function registry with all Phase 1 built-in functions."""
    reg = FunctionRegistry()

    # Character functions
    from saslite.functions import char_funcs
    reg.register("SUBSTR", char_funcs.substr)
    reg.register("SCAN", char_funcs.scan)
    reg.register("COMPRESS", char_funcs.compress)
    reg.register("UPCASE", char_funcs.upcase)
    reg.register("LOWCASE", char_funcs.lowcase)
    reg.register("STRIP", char_funcs.strip)
    reg.register("TRIM", char_funcs.trim)
    reg.register("LEFT", char_funcs.left)
    reg.register("CAT", char_funcs.cat)
    reg.register("CATS", char_funcs.cats)
    reg.register("CATX", char_funcs.catx)
    reg.register("COMPBL", char_funcs.compbl)
    reg.register("TRANWRD", char_funcs.tranwrd)
    reg.register("INDEX", char_funcs.index)
    reg.register("FIND", char_funcs.find)
    reg.register("COUNT", char_funcs.count)
    reg.register("REPEAT", char_funcs.repeat)
    reg.register("REVERSE", char_funcs.reverse)
    reg.register("LENGTH", char_funcs.length)
    reg.register("LENGTHC", char_funcs.lengthc)
    reg.register("MISSING", char_funcs.missing)
    reg.register("COALESCEC", char_funcs.coalescec)
    reg.register("PRXMATCH", char_funcs.prxmatch)
    reg.register("PROPCASE", char_funcs.propcase)
    reg.register("COUNTW", char_funcs.countw)
    reg.register("VERIFY", char_funcs.verify)
    reg.register("SUBSTRN", char_funcs.substrn)
    reg.register("TRANSLATE", char_funcs.translate)
    reg.register("LIKE", char_funcs.like_match)
    reg.register("RANK", char_funcs.rank_char)
    reg.register("BYTE", char_funcs.byte)

    # Numeric functions
    from saslite.functions import numeric_funcs
    reg.register("SUM", numeric_funcs.sum)
    reg.register("MEAN", numeric_funcs.mean)
    reg.register("MIN", numeric_funcs.min_val)
    reg.register("MAX", numeric_funcs.max_val)
    reg.register("N", numeric_funcs.n)
    reg.register("NMISS", numeric_funcs.nmiss)
    reg.register("ROUND", numeric_funcs.round_val)
    reg.register("INT", numeric_funcs.int_val)
    reg.register("MOD", numeric_funcs.mod_val)
    reg.register("CEIL", numeric_funcs.ceil_val)
    reg.register("FLOOR", numeric_funcs.floor_val)
    reg.register("ABS", numeric_funcs.abs_val)
    reg.register("SQRT", numeric_funcs.sqrt_val)
    reg.register("LOG", numeric_funcs.log_val)
    reg.register("LOG10", numeric_funcs.log10_val)
    reg.register("EXP", numeric_funcs.exp_val)
    reg.register("SIN", numeric_funcs.sin_val)
    reg.register("COS", numeric_funcs.cos_val)
    reg.register("TAN", numeric_funcs.tan_val)
    reg.register("SIGN", numeric_funcs.sign_val)
    reg.register("STD", numeric_funcs.std_val)
    reg.register("RANGE", numeric_funcs.range_val)
    reg.register("MEDIAN", numeric_funcs.median_val)
    reg.register("RANUNI", numeric_funcs.ranuni)
    reg.register("UNIFORM", numeric_funcs.ranuni)
    reg.register("RANNOR", numeric_funcs.rannor)
    reg.register("NORMAL", numeric_funcs.rannor)
    reg.register("PROBNORM", numeric_funcs.probnorm)
    reg.register("PROBIT", numeric_funcs.probit)
    reg.register("PROBT", numeric_funcs.probt)
    reg.register("PROBF", numeric_funcs.probf)
    reg.register("PROBCHI", numeric_funcs.probchi)

    # Date functions
    from saslite.functions import date_funcs
    reg.register("TODAY", date_funcs.today)
    reg.register("DATE", date_funcs.date_val)
    reg.register("DATETIME", date_funcs.datetime_val)
    reg.register("MDY", date_funcs.mdy)
    reg.register("YEAR", date_funcs.year)
    reg.register("MONTH", date_funcs.month)
    reg.register("DAY", date_funcs.day)
    reg.register("WEEKDAY", date_funcs.weekday)
    reg.register("QTR", date_funcs.qtr)
    reg.register("INTNX", date_funcs.intnx)
    reg.register("INTCK", date_funcs.intck)
    reg.register("DATEPART", date_funcs.datepart)
    reg.register("TIMEPART", date_funcs.timepart)
    reg.register("DATEDIF", date_funcs.datedif)
    reg.register("HOUR", date_funcs.hour)
    reg.register("MINUTE", date_funcs.minute)

    # Conversion functions
    from saslite.functions import convert_funcs
    reg.register("INPUT", convert_funcs.input_sas)
    reg.register("PUT", convert_funcs.put_sas)
    reg.register("PUTN", convert_funcs.putn)
    reg.register("PUTC", convert_funcs.putc)
    reg.register("INPUTN", convert_funcs.inputn)

    # Conditional functions
    from saslite.functions import conditional_funcs
    reg.register("IFC", conditional_funcs.ifc)
    reg.register("IFN", conditional_funcs.ifn)
    reg.register("COALESCE", conditional_funcs.coalesce_num)

    return reg
