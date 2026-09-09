"""SAS numeric functions."""

from __future__ import annotations

import math
import statistics
from typing import Any

from saslite.runtime.types import is_missing


def sum(*args: Any) -> float:
    """SUM(n1, n2, ...) — sum, ignoring missing values. Returns 0 if no valid values."""
    vals = [float(v) for v in args if not is_missing(v)]
    if not vals:
        return 0.0
    return math.fsum(vals)


def mean(*args: Any) -> float:
    """MEAN(n1, n2, ...) — arithmetic mean, ignoring missing."""
    vals = [float(v) for v in args if not is_missing(v)]
    if not vals:
        return float("nan")
    return math.fsum(vals) / len(vals)  # type: ignore


def min_val(*args: Any) -> float:
    """MIN(n1, n2, ...) — minimum, ignoring missing."""
    vals = [float(v) for v in args if not is_missing(v)]
    if not vals:
        return float("nan")
    return min(vals)


def max_val(*args: Any) -> float:
    """MAX(n1, n2, ...) — maximum, ignoring missing."""
    vals = [float(v) for v in args if not is_missing(v)]
    if not vals:
        return float("nan")
    return max(vals)


def n(*args: Any) -> int:
    """N(n1, n2, ...) — count of non-missing values."""
    return len([v for v in args if not is_missing(v)])


def nmiss(*args: Any) -> int:
    """NMISS(n1, n2, ...) — count of missing values."""
    return len([v for v in args if is_missing(v)])


def round_val(number: Any, unit: float = 1.0) -> float:
    """ROUND(number [, unit]) — round to nearest unit using SAS rounding (away from zero for .5)."""
    if is_missing(number):
        return float("nan")
    number = float(number)
    # Handle infinity
    if math.isinf(number):
        return number
    if unit == 0:
        return number
    # SAS uses "round half away from zero" (traditional rounding)
    # Python's round() uses "round half to even" (banker's rounding)
    scaled = number / unit
    if scaled >= 0:
        result = math.floor(scaled + 0.5)
    else:
        result = math.ceil(scaled - 0.5)
    return result * unit


def int_val(number: Any) -> float:
    """INT(number) — integer part (truncate toward zero)."""
    if is_missing(number):
        return float("nan")
    return float(int(float(number)))


def mod_val(dividend: Any, divisor: Any) -> float:
    """MOD(dividend, divisor) — modulo."""
    if is_missing(dividend) or is_missing(divisor):
        return float("nan")
    d = float(divisor)
    if d == 0:
        return float("nan")
    return float(dividend) % d


def ceil_val(number: Any) -> float:
    """CEIL(number) — ceiling."""
    if is_missing(number):
        return float("nan")
    return math.ceil(float(number))


def floor_val(number: Any) -> float:
    """FLOOR(number) — floor."""
    if is_missing(number):
        return float("nan")
    return math.floor(float(number))


def abs_val(number: Any) -> float:
    """ABS(number) — absolute value."""
    if is_missing(number):
        return float("nan")
    return abs(float(number))


def sqrt_val(number: Any) -> float:
    """SQRT(number) — square root."""
    if is_missing(number):
        return float("nan")
    val = float(number)
    if val < 0:
        return float("nan")
    return math.sqrt(val)


def log_val(number: Any) -> float:
    """LOG(number) — natural log."""
    if is_missing(number):
        return float("nan")
    val = float(number)
    if val <= 0:
        return float("nan")
    return math.log(val)


def log10_val(number: Any) -> float:
    """LOG10(number) — log base 10."""
    if is_missing(number):
        return float("nan")
    val = float(number)
    if val <= 0:
        return float("nan")
    return math.log10(val)


def exp_val(number: Any) -> float:
    """EXP(number) — e^number."""
    if is_missing(number):
        return float("nan")
    return math.exp(float(number))


def sin_val(number: Any) -> float:
    """SIN(number)."""
    if is_missing(number):
        return float("nan")
    return math.sin(float(number))


def cos_val(number: Any) -> float:
    """COS(number)."""
    if is_missing(number):
        return float("nan")
    return math.cos(float(number))


def tan_val(number: Any) -> float:
    """TAN(number)."""
    if is_missing(number):
        return float("nan")
    return math.tan(float(number))


def sign_val(number: Any) -> float:
    """SIGN(number) — -1, 0, or 1."""
    if is_missing(number):
        return float("nan")
    val = float(number)
    if val > 0:
        return 1.0
    if val < 0:
        return -1.0
    return 0.0


def std_val(*args: Any) -> float:
    """STD(n1, n2, ...) — standard deviation."""
    vals = [float(v) for v in args if not is_missing(v)]
    if len(vals) < 2:
        return float("nan")
    return statistics.stdev(vals)


def range_val(*args: Any) -> float:
    """RANGE(n1, n2, ...) — max - min."""
    vals = [float(v) for v in args if not is_missing(v)]
    if not vals:
        return float("nan")
    return max(vals) - min(vals)


def median_val(*args: Any) -> float:
    """MEDIAN(n1, n2, ...) — median value."""
    vals = sorted(float(v) for v in args if not is_missing(v))
    if not vals:
        return float("nan")
    return statistics.median(vals)


def missing_num(v: Any) -> int:
    """MISSING(numeric_var) — test for missing (1=missing)."""
    return 1 if is_missing(v) else 0


def coalesce_num(*args: Any) -> float:
    """COALESCE(n1, n2, ...) — first non-missing numeric."""
    for a in args:
        if not is_missing(a):
            return float(a)
    return float("nan")


# ─── Random number functions ─────────────────────────

import random as _random

_rng = _random.Random()
_rng_seeded = False


def _ensure_seed(seed: Any) -> None:
    """Seed the shared RNG on first call with a positive seed (SAS semantics)."""
    global _rng_seeded
    if not _rng_seeded:
        try:
            s = int(float(seed))
        except (TypeError, ValueError):
            s = 0
        if s > 0:
            _rng.seed(s)
        _rng_seeded = True


def ranuni(seed: Any = 0) -> float:
    """RANUNI(seed) — uniform random number in [0, 1)."""
    _ensure_seed(seed)
    return _rng.random()


def rannor(seed: Any = 0) -> float:
    """RANNOR(seed) — standard normal random number."""
    _ensure_seed(seed)
    return _rng.gauss(0.0, 1.0)


# ─── Probability distribution functions ──────────────


def probnorm(x: Any) -> float:
    """PROBNORM(x) — standard normal CDF."""
    if is_missing(x):
        return float("nan")
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def probit(p: Any) -> float:
    """PROBIT(p) — standard normal quantile (inverse CDF).

    Acklam's rational approximation (relative error < 1.15e-9).
    """
    if is_missing(p):
        return float("nan")
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return float("nan")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))
    q = math.sqrt(-2 * math.log(1 - p))
    return -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
             / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    max_iter = 300
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _gammainc_lower(s: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(s, x)."""
    if x < 0 or s <= 0:
        return float("nan")
    if x == 0:
        return 0.0
    if x < s + 1.0:
        # Series representation
        term = 1.0 / s
        total = term
        n = s
        for _ in range(300):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 3e-14:
                break
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    # Continued fraction for Q(s, x), then P = 1 - Q
    fpmin = 1e-300
    b = x + 1.0 - s
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 300):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < 3e-14:
            break
    q = math.exp(-x + s * math.log(x) - math.lgamma(s)) * h
    return 1.0 - q


def probt(x: Any, df: Any) -> float:
    """PROBT(x, df) — Student's t distribution CDF."""
    if is_missing(x) or is_missing(df):
        return float("nan")
    x = float(x)
    df = float(df)
    if df <= 0:
        return float("nan")
    t2 = x * x
    ib = _betainc(df / 2.0, 0.5, df / (df + t2))
    if x > 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def probf(x: Any, ndf: Any, ddf: Any) -> float:
    """PROBF(x, ndf, ddf) — F distribution CDF."""
    if is_missing(x) or is_missing(ndf) or is_missing(ddf):
        return float("nan")
    x = float(x)
    ndf = float(ndf)
    ddf = float(ddf)
    if x <= 0 or ndf <= 0 or ddf <= 0:
        return 0.0 if x <= 0 else float("nan")
    return _betainc(ndf / 2.0, ddf / 2.0, ndf * x / (ndf * x + ddf))


def probchi(x: Any, df: Any) -> float:
    """PROBCHI(x, df) — chi-square distribution CDF."""
    if is_missing(x) or is_missing(df):
        return float("nan")
    x = float(x)
    df = float(df)
    if x <= 0 or df <= 0:
        return 0.0 if x <= 0 else float("nan")
    return _gammainc_lower(df / 2.0, x / 2.0)
