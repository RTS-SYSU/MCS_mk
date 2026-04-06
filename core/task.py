from typing import Optional


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
                 utility: float = 1.0):

        self.id = id
        self.criticality = criticality
        self.period = period
        self.deadline = deadline
        self.wcet_lo = wcet_lo
        self.wcet_hi = wcet_hi
        self.utility = utility

        self.is_backup_subblock = False  # 是否为仅在 HI 模式运行的子块
        self.parent_id = None  # 用于追踪原任务ID

        if criticality == "LO":
            self.mk: MKPattern = MKPattern(m, k, offset=0)
            #self.mk
        else:
            self.mk: MKPattern = MKPattern(k, k)  # HI tasks: always mandatory

        self.priority: Optional[int] = None  # lower priority value, higher priority

    # def increase_x(self, num: int):
    #     """Increase only the effective m (m+x), but DO NOT modify original m."""
    #     if self.mk:
    #         self.mk.increase_x(num)

    def get_pattern(self):
        return self.mk.get_pattern() if self.mk else None

    def __repr__(self):
        tag = "[SUB]" if self.is_backup_subblock else ""
        return (f"Task[{tag}](id={self.id}, criticality={self.criticality}, period={self.period},"
                f"deadline={self.deadline}, wcet_lo={self.wcet_lo}, wcet_hi={self.wcet_hi},"
                f"utility={self.utility}, priority={self.priority}, mk={self.mk})")


class MKPattern:
    """
    Represent (m,k)-pattern.
    m = base mandatory jobs (fixed)
    x = dynamic increment applied by offline algorithm
    effective_m = m + x
    """

    def __init__(self, m: int, k: int, offset: int = 0):

        if m > k or m < 0:
            raise ValueError(f"Invalid m-k: m={m}, k={k}")

        self.m = m  # base m (fixed)
        self.k = k
        self.x: int = 0  # dynamic increment
        self.dx: int = 0  # degraded x
        self.offset = offset  # 新增：模式偏移量

        self.pattern = []
        self.pattern_degrade = []
        # initial pattern
        self.update_pattern()
        self.update_pattern(is_degraded=True)

    # def _update_pattern(self):
    #     effective_m = min(self.m + self.x, self.k)
    #     self.pattern = [1] * effective_m + [0] * (self.k - effective_m)

    def update_pattern(self, is_degraded: bool = False):
        if not is_degraded:
            # 考虑偏移量和优化后的 x
            effective_m = min(self.m + self.x, self.k)
            # 生成一个循环模式，从 offset 开始填充 effective_m 个 1
            p = [0] * self.k
            for i in range(effective_m):
                p[(self.offset + i) % self.k] = 1
            self.pattern = p
        else:
            effective_m = min(self.m + self.dx, self.k)
            # 生成一个循环模式，从 offset 开始填充 effective_m 个 1
            p = [0] * self.k
            for i in range(effective_m):
                p[(self.offset + i) % self.k] = 1
            self.pattern_degrade = p

    def increase_x(self, num: int, is_degraded: bool = False):
        """
        Apply increase or decrease to x (not m). m stays unchanged forever. But x must remain >= 0.

        param: num is the increment (positive or negative).

        """
        if not isinstance(num, int):
            return  # ignore invalid inputs
        if not is_degraded:
            # update x
            self.x += num
            # ensure x >= 0
            if self.x < 0:
                self.x = 0
            self.update_pattern(is_degraded=is_degraded)
        else:
            self.dx += num
            # ensure x >= 0
            if self.dx < 0:
                self.dx = 0
            self.update_pattern(is_degraded=is_degraded)

    def reset_x(self, is_degraded: bool = False):
        """Reset dynamic increment (for next window or offline reset)."""
        if not is_degraded:
            self.x = 0
            self.update_pattern(is_degraded=is_degraded)
        else:
            self.dx = 0
            self.update_pattern(is_degraded=is_degraded)

    def get_pattern(self):
        return self.pattern

    def merge_pattern(self, other: 'MKPattern') -> 'MKPattern':
        """
        将另一个子块的 pattern 按位 OR 合并，返回一个新的 MKPattern。
        合并不改变原有 '1' 的位置，只是叠加新的 '1'。
        例如: (0,1,0,0) merge (0,0,1,0) -> (0,1,1,0)
        """
        # if self.k != other.k:
        #     raise ValueError(
        #         f"Cannot merge patterns with different k: {self.k} vs {other.k}"
        #     )
        # 按位 OR 合并
        merged_pattern = [a | b for a, b in zip(self.pattern, other.pattern)]
        new_m = sum(merged_pattern)

        # 创建新的 MKPattern，m 直接等于合并后 1 的数量
        new_mk = MKPattern(m=new_m, k=self.k)

        # 直接覆盖 pattern 为合并结果（绕过 offset 的影响）
        new_mk.pattern = merged_pattern
        new_mk.pattern_degrade = merged_pattern

        return new_mk

    def __repr__(self):
        return f"MKPattern(m={self.m}, x={self.x}, k={self.k}, pattern={self.pattern})"
