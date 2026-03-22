from typing import List

from core.task import Task


class Processor:
    """
    A processor core
    """
    def __init__(self, pid: int):
        self.id = pid
        self.tasks: List[Task] = []
        self.drop_list: List[Task] = [] # 该核心上需要在HI模式丢弃的任务
        self.utilization_lo: float = 0.0
        self.utilization_hi: float = 0.0

    def add_task(self, task: Task):
        self.tasks.append(task)
        if task.criticality == "HI":
            self.utilization_lo += (task.wcet_lo / task.period)
            self.utilization_hi += (task.wcet_hi / task.period)
        else:
            eff_factor = task.mk.m / task.mk.k
            if task.is_backup_subblock:
                self.utilization_hi += (task.wcet_hi / task.period) * eff_factor
            else:
                self.utilization_lo += (task.wcet_lo / task.period) * eff_factor
                self.utilization_hi += (task.wcet_hi / task.period) * eff_factor


    def remove_task(self, task: Task):
        if task in self.tasks:
            self.tasks.remove(task)
            self._update_util(task, sign=-1)
            # 如果在 drop list 中也要移除
            if task in self.drop_list:
                self.drop_list.remove(task)

    def mark_as_dropped(self, task: Task):
        """
        标记任务在 HI 模式下丢弃 (LO 模式继续运行)。
        动作：加入 drop_list，并扣除其 HI 模式的利用率贡献。
        """
        if task in self.tasks and task not in self.drop_list:
            self.drop_list.append(task)

            # 扣除 HI 模式利用率 (LO 模式保持不变)
            # 只有 LO 任务会被 Drop
            if task.criticality == "LO":
                eff_factor = task.mk.m / task.mk.k
                u_dec = (task.wcet_lo / task.period) * eff_factor
                self.utilization_hi -= u_dec

    def _update_util(self, task: Task, sign: int = 1):
        if task.criticality == "HI":
            # HI 任务
            self.utilization_lo += sign * (task.wcet_lo / task.period)
            self.utilization_hi += sign * (task.wcet_hi / task.period)
        else:
            # LO 任务
            eff_factor = task.mk.m / task.mk.k
            u_inc = (task.wcet_lo / task.period) * eff_factor

            if task.is_backup_subblock:
                # 【关键逻辑】备份子块：仅在 HI 模式运行
                # LO 利用率不增加 (因为 LO 模式下它休眠)
                self.utilization_hi += sign * u_inc
            else:
                # 普通 LO 任务 (包括未被 Drop 的原任务)
                self.utilization_lo += sign * u_inc
                # 如果不在 drop_list 中，它也贡献 HI 利用率
                # (为简化计算，add时默认贡献，之后通过 mark_as_dropped 扣除)
                self.utilization_hi += sign * u_inc
    def __repr__(self):
        return f"Processor(id={self.id}, tasks_count={len(self.tasks)}, U_LO={self.utilization_lo:.2f})"



