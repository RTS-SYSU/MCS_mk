from typing import List

from core.task import Task


class Processor:
    """
    A processor core that hosts a set of partitioned tasks.
    """

    def __init__(self, pid: int):
        self.id = pid
        self.tasks: List[Task] = []
        self.drop_list: List[Task] = []  # \overline{\Gamma}_{LO}^{*}: suspended in mode switch
        self.utilization_lo: float = 0.0
        self.utilization_hi: float = 0.0

    def add_task(self, task: Task):
        self.tasks.append(task)
        u_inc = (task.wcet_lo / task.period) * (task.mk.m / task.mk.k)
        if task.criticality == "HI":
            self.utilization_lo += task.wcet_lo / task.period
            self.utilization_hi += task.wcet_hi / task.period
        else:
            self.utilization_lo += u_inc
            self.utilization_hi += u_inc

    def remove_task(self, task: Task):
        if task in self.tasks:
            self.tasks.remove(task)
            self._update_util(task, sign=-1)
        if task in self.drop_list:
            self.drop_list.remove(task)

    def mark_as_dropped(self, task: Task):
        """
        Mark a LO task to be suspended during mode switch and HI mode.
        Deducts its HI-mode utilization contribution.
        """
        if task in self.tasks and task not in self.drop_list:
            self.drop_list.append(task)
            if task.criticality == "LO":
                eff_factor = task.mk.m / task.mk.k
                u_dec = (task.wcet_lo / task.period) * eff_factor
                self.utilization_hi -= u_dec

    def _update_util(self, task: Task, sign: int = 1):
        eff_factor = task.mk.m / task.mk.k
        if task.criticality == "HI":
            self.utilization_lo += sign * (task.wcet_lo / task.period)
            self.utilization_hi += sign * (task.wcet_hi / task.period)
        else:
            u_delta = (task.wcet_lo / task.period) * eff_factor
            self.utilization_lo += sign * u_delta
            self.utilization_hi += sign * u_delta

    def __repr__(self):
        return (f"Processor(id={self.id}, tasks_count={len(self.tasks)}, "
                f"U_LO={self.utilization_lo:.2f}, U_HI={self.utilization_hi:.2f})")
