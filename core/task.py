from typing import Optional, Literal

Mode = Literal['L', 'S', 'H']  # L: LO mode, S: mode Switch, H: stable HI mode


class Task:

    def __init__(self,
                 id: int,
                 criticality: str,  # 'HI' or 'LO'
                 period: float,
                 deadline: float,
                 wcet_lo: float,
                 wcet_hi: float,
                 m: int = 1,
                 k: int = 1,
                 baseline_importance: float = 1.0):

        self.id = id
        self.criticality = criticality
        self.period = period
        self.deadline = deadline
        self.wcet_lo = wcet_lo
        self.wcet_hi = wcet_hi
        self.baseline_importance = baseline_importance  # μ_i

        if criticality == "LO":
            self.mk: MKPattern = MKPattern(m, k, offset=0)
        else:
            self.mk: MKPattern = MKPattern(k, k)  # HI tasks: always mandatory

        self.priority: Optional[int] = None  # smaller value = higher priority

    def calculate_importance(self, beta: float, mode: Mode = 'L') -> float:
        """
        I_i(x_i) = μ_i * (1 + β * x_i / (k_i - m_i))

        mode: 'L' (x^L), 'S' (x^S), 'H' (x^H)
        For HI tasks (k_i == m_i): x_i = 0, returns μ_i.
        """
        if self.mk is None:
            return 0.0
        m_i = self.mk.m
        k_i = self.mk.k
        if k_i == m_i:
            return self.baseline_importance
        x = self.mk.get_x(mode)
        return self.baseline_importance * (1.0 + beta * x / (k_i - m_i))

    def get_pattern(self, mode: Mode = 'L'):
        return self.mk.get_pattern(mode) if self.mk else None

    def __repr__(self):
        return (f"Task(id={self.id}, criticality={self.criticality}, period={self.period},"
                f"deadline={self.deadline}, wcet_lo={self.wcet_lo}, wcet_hi={self.wcet_hi},"
                f"μ={self.baseline_importance}, priority={self.priority}, mk={self.mk})")


class MKPattern:
    """
    (m,k) weakly-hard constraint with three mode-specific augmented levels.

    m = base mandatory jobs (fixed)
    x_l = augmented level in LO mode        (x^L)
    x_s = degraded augmented level in mode switch (x^S)
    x_h = augmented level in stable HI mode (x^H)

    effective_m = m + x_mode, clamped to [0, k].
    """

    def __init__(self, m: int, k: int, offset: int = 0):

        if m > k or m < 0:
            raise ValueError(f"Invalid m-k: m={m}, k={k}")

        self.m = m
        self.k = k
        self.x_l: int = 0
        self.x_s: int = 0
        self.x_h: int = 0
        self.offset = offset

        self._pattern_l: list[int] = []
        self._pattern_s: list[int] = []
        self._pattern_h: list[int] = []
        self._update_all_patterns()

    def _update_all_patterns(self):
        self._pattern_l = self._build_pattern(self.x_l)
        self._pattern_s = self._build_pattern(self.x_s)
        self._pattern_h = self._build_pattern(self.x_h)

    def _build_pattern(self, x: int) -> list[int]:
        effective_m = max(0, min(self.m + x, self.k))
        p = [0] * self.k
        for i in range(effective_m):
            p[(self.offset + i) % self.k] = 1
        return p

    def get_x(self, mode: Mode = 'L') -> int:
        if mode == 'L':
            return self.x_l
        elif mode == 'S':
            return self.x_s
        else:
            return self.x_h

    def set_x(self, value: int, mode: Mode = 'L'):
        """Set augmented level for a given mode."""
        if mode == 'L':
            self.x_l = max(0, min(value, self.k - self.m))
        elif mode == 'S':
            self.x_s = max(0, min(value, self.k - self.m))
        else:
            self.x_h = max(0, min(value, self.k - self.m))
        self._update_all_patterns()

    def increase_x(self, delta: int, mode: Mode = 'L'):
        """Apply a delta to augmented level for a given mode (positive = increase, negative = decrease)."""
        new_val = self.get_x(mode) + delta
        self.set_x(new_val, mode)

    def reset_x(self, mode: Mode = 'L'):
        self.set_x(0, mode)

    def get_effective_m(self, mode: Mode = 'L') -> int:
        return self.m + self.get_x(mode)

    def get_pattern(self, mode: Mode = 'L') -> list[int]:
        if mode == 'L':
            return self._pattern_l
        elif mode == 'S':
            return self._pattern_s
        else:
            return self._pattern_h

    def merge_pattern(self, other: 'MKPattern') -> 'MKPattern':
        """Bitwise-OR merge of two patterns (from sub-blocks)."""
        merged = [a | b for a, b in zip(self._pattern_l, other._pattern_l)]
        new_m = sum(merged)
        new_mk = MKPattern(m=new_m, k=self.k)
        new_mk._pattern_l = merged
        new_mk._pattern_s = merged
        new_mk._pattern_h = merged
        return new_mk

    def __repr__(self):
        return (f"MKPattern(m={self.m}, k={self.k}, "
                f"x_l={self.x_l}, x_s={self.x_s}, x_h={self.x_h}, "
                f"pattern_L={self._pattern_l})")
