import logging
import math
import copy
import heapq  # for efficiently managing jobs that have not yet arrived
import random
from typing import List, Dict, Optional, Tuple, Any

from core.job import Job
from core.processor import Processor
from core.task import Task

# Constant representing the lowest priority level for Optional Jobs.
# In Fixed Priority Scheduling, larger numbers indicate lower priority.
OPTIONAL_PRIORITY_START = 9999


class OnlineSimulator:
    def __init__(self, tasks: List[Task], drop_list: List[Task], mode_switch_time: float = float('inf')):
        """
        :param tasks: Task set
        :param drop_list: Tasks to drop in HI mode
        :param mode_switch_time: The exact time to switch from LO to HI.
                0.0 = Stable HI; inf = Stable LO; >0 = Dynamic switch.
        """
        # protect task: Deep copy to ensure internal state changes do not affect external objects
        self.tasks = copy.deepcopy(tasks)

        # For easy to find, build a mapping
        self.task_map = {t.id: t for t in self.tasks}

        self.drop_list_ids = {t.id for t in drop_list}
        # print(drop_list)
        # print(self.drop_list_ids)
        self.current_time = 0.0

        # Initialize System Mode based on switch time
        self.mode_switch_time = float(mode_switch_time)
        self.mode = 'HI' if self.mode_switch_time <= 0 else 'LO'

        # Runtime queues
        self.ready_queue: List[Job] = []
        self.current_job: Optional[Job] = None
        self.optional_prio_counter = OPTIONAL_PRIORITY_START

        # Statistics container
        self.stats = {
            "total_released": 0,
            "lo_released_count": 0,
            "lo_completed_count": 0,
            "mandatory_completed": 0,
            "optional_completed": 0,
            "optional_released": 0,
            "deadline_misses": 0,
            "utility": 0.0,
            "weighted_released": 0.0,
            "weighted_completed": 0.0,
        }

        # Discrete Event Queue: (timestamp, event_type, task_index)
        # event_type: 0 = Mode Switch, 1 = Job Arrival
        self.events: List[Tuple[float, int, int]] = []

        # Schedule first arrival for all tasks
        for i, task in enumerate(self.tasks):
            heapq.heappush(self.events, (0.0, 1, i))

        if 0 < self.mode_switch_time < float('inf'):
            heapq.heappush(self.events, (self.mode_switch_time, 0, -1))

        self.job_seq_counters = {t.id: 0 for t in self.tasks}

        # [LOGGING] Initialize logger instance
        self.logger = logging.getLogger("Sim")

    def _get_pattern_bit(self, task: Task, job_seq: int) -> int:
        """
        Dynamically determine if the job is Mandatory (1) or Optional (0).
        Logic depends on current Mode.
        """
        if not task.mk: return 1

        # 如果是备份块，它始终是 (1, k)，没有 x 优化
        if task.is_backup_subblock:
            effective_m = task.mk.m
        elif self.mode == 'HI':
            # 原任务在 HI 模式：如果在 drop list，它根本不应该运行(admission 会拦截)
            # 如果不在 drop list，说明它是幸存者，退回 m
            effective_m = task.mk.m
        else:
            # LO 模式：享受 x
            effective_m = task.mk.m + task.mk.x

        k = task.mk.k
        offset = task.mk.offset
        idx = job_seq % k
        normalized_idx = (idx - offset) % k

        return 1 if normalized_idx < effective_m else 0

    def _calculate_interference_and_slack(self, job: Job) -> float:
        """
            Calculates the available slack for an Optional job (lowest priority) within [current_time, deadline].
            In this paper, current time is equal to the released time or decision time of job.

            Formula: Slack = Interval_Length - Future_Interference - Current_Interference.

            [SAFETY CRITICAL]:
            When calculating interference, we MUST use the WCET (Budget), not ACET.
            We cannot predict if future or queued jobs will finish early, so we assume the worst case.
        """
        interval_len = job.absolute_deadline - self.current_time
        # interference
        inter_hi = 0.0
        inter_lo = 0.0
        inter_queue = 0.0
        inter_curr = 0.0


        for task in self.tasks:
            # Safety Rule: Always assume HI tasks execute for WCET_HI to guarantee safety
            # in case of a sudden mode switch.
            # 1. HI Tasks Interference
            if task.criticality == "HI":
                # Find the very next arrival time (>= current_time)
                # Assuming synchronous release at t=0
                next_arrival = math.ceil(self.current_time / task.period) * task.period
                # Iterate through all future arrivals within the interval
                while next_arrival < job.absolute_deadline:
                    inter_hi += task.wcet_hi
                    next_arrival += task.period

            # 2. LO Mandatory Interference
            # 这里存在一个问题，无法确定何时发生模式转换，因此得删去，否则会导致一个optional job 被执行，但是miss deadline. 3/22
            # elif task.criticality == "LO" and task.id != job.task_id:
            #     # Filter based on current mode policies
            #     if task.is_backup_subblock:
            #         if self.mode == 'LO': continue  # Backups don't run in LO
            #     elif self.mode == 'HI' and task.id in self.drop_list_ids:
            #         continue  # Dropped tasks don't run in HI

                # Calculate future mandatory arrivals
                start_k = math.ceil(self.current_time / task.period)
                end_time = job.absolute_deadline
                k = start_k
                while k * task.period < end_time:
                    if self._get_pattern_bit(task, k) == 1:
                        # Use correct WCET
                        wcet = task.wcet_lo #   task.wcet_hi == task.wcet_lo
                        inter_lo += wcet
                    k += 1

        # 3. Queue Interference (Existing Jobs)
        for q_job in self.ready_queue:
            #  q_job.priority < job.priority must be met, job.priority is lowest.
            if q_job.priority < job.priority:
                # Use 'remaining_budget' (WCET), NOT 'remaining_time' (ACET).
                # The scheduler does not know the queued job will finish early.
                inter_queue += q_job.remaining_budget

        # 4. Current Job Interference
        if self.current_job and self.current_job != job:
            inter_curr += self.current_job.remaining_budget

        total_interference = inter_hi + inter_lo + inter_queue + inter_curr
        slack = interval_len - total_interference

        # [LOGGING] Output detailed Slack Calculation logic
        self.logger.debug(
            f"   >>> [Slack Test Details] Target: T{job.task_id}-{job.job_id} | Time: {self.current_time:.2f} | Deadline: {job.absolute_deadline:.2f}\n"
            f"       Interval: {interval_len:.2f}\n"
            f"       Interference Breakdown: HI_Fut={inter_hi:.2f}, LO_Fut={inter_lo:.2f}, Queue={inter_queue:.2f}, Curr={inter_curr:.2f}\n"
            f"       Total Inter: {total_interference:.2f} | Slack: {slack:.2f} | Required Budget: {job.initial_wcet:.2f}"
        )

        return slack

    def _handle_mode_switch_to_hi(self):
        """
        Handles system transition from LO to HI mode.
        1. Expands WCET for HI tasks (LO WCET -> HI WCET).
        2. Drops LO tasks listed in the drop_list.
        """
        self.mode = 'HI'

        # 1. Update Current Job
        if self.current_job:
            if self.current_job.criticality == "HI":
                # Find task definition to get WCET diff (HI WCET - LO WCET)
                task = self.tasks[next(i for i, t in enumerate(self.tasks) if t.id == self.current_job.task_id)]
                delta = task.wcet_hi - task.wcet_lo
                # Extend both Physical time and Logical budget
                self.current_job.remaining_time += delta
                # ratio = task.wcet_hi / task.wcet_lo
                # self.current_job.remaining_time *= ratio # another way
                self.current_job.remaining_budget += delta
                self.current_job.initial_wcet = task.wcet_hi
                # [LOGGING]
                self.logger.info(f"    -> Expanding Current Job T{self.current_job.task_id} by {delta:.2f}")

            elif self.current_job.task_id in self.drop_list_ids:
                # [LOGGING]
                self.logger.info(f"    -> Dropping Current Job T{self.current_job.task_id} (Drop List)")
                self.current_job = None  # Drop immediately

        # 2. Update Ready Queue
        new_queue = []
        for job in self.ready_queue:
            # Drop jobs from drop_list
            if job.task_id in self.drop_list_ids:
                # [LOGGING]
                self.logger.info(f"    -> Dropping Queued Job T{job.task_id}-{job.job_id}")
                continue

            # Expand budgets for HI tasks
            if job.criticality == "HI":
                task = self.tasks[next(i for i, t in enumerate(self.tasks) if t.id == job.task_id)]
                delta = task.wcet_hi - task.wcet_lo
                job.remaining_time += delta
                job.remaining_budget += delta
                job.initial_wcet = task.wcet_hi

            new_queue.append(job)

        self.ready_queue = new_queue
        # Re-sort to maintain priority order
        self.ready_queue.sort(key=lambda j: (j.priority, j.arrival_time))

    def run(self, duration: float) -> Dict:
        """
        Main Discrete Event Simulation Loop.
        """

        # [LOGGING] Start
        self.logger.info(f"--- Simulation Start (Duration: {duration}, Initial Mode: {self.mode}) ---")
        while self.current_time < duration:
            # Next Event Time Calculation
            next_event_time = self.events[0][0] if self.events else float('inf')
            next_completion = (self.current_time + self.current_job.remaining_time) if self.current_job else float(
                'inf')

            # Jump to the nearest critical moment (Arrival, Mode Switch, or Completion)
            next_critical_time = min(next_event_time, next_completion)

            if next_critical_time > duration:
                if self.current_job:
                    self.current_job.remaining_time -= (duration - self.current_time)
                break

            time_step = next_critical_time - self.current_time
            self.current_time = next_critical_time

            #  Execution Update
            if self.current_job:
                # Decrement ACET and WCET
                self.current_job.remaining_time -= time_step
                self.current_job.remaining_budget -= time_step

                # If remaining_time <= 0, the job leaves the CPU early.
                # The remaining_budget might still be > 0, which effectively becomes slack because this job stops interfering with others.
                if self.current_job.remaining_time <= 1e-9:
                    self.current_job.finish_time = self.current_time
                    if self.current_job.criticality == "LO":
                        self.stats["lo_completed_count"] += 1

                        # add utility from completed LO task
                        task_util = self.task_map[self.current_job.task_id].utility
                        self.stats["weighted_completed"] += task_util

                    # [LOGGING] Job Finish
                    self.logger.info(
                        f"[{self.current_time:.2f}] [Finish] Job T{self.current_job.task_id}-{self.current_job.job_id} completed.")
                    # It is impossible to happen in online phase because the task set passes schedulability test.
                    if self.current_job.finish_time > self.current_job.absolute_deadline:
                        self.stats["deadline_misses"] += 1
                        # [LOGGING] Error
                        # self.logger.error(
                        #     f"    !!! Deadline Miss !!! Job T{self.current_job.task_id}-{self.current_job.job_id}")
                        miss_type = "OPT" if self.current_job.is_optional else "MAND"
                        self.logger.error(
                            f"!!! DEADLINE MISS !!! Job T{self.current_job.task_id}-{self.current_job.job_id} ({miss_type})\n"
                            f"    Finish: {self.current_job.finish_time} > Deadline: {self.current_job.absolute_deadline}\n"
                            f"    Current Mode: {self.mode}\n"
                            f"    WCET: {self.current_job.initial_wcet}, ACET: {self.current_job.remaining_time + time_step}"
                        )
                    if self.current_job.is_optional:
                        self.stats["optional_completed"] += 1
                    else:
                        self.stats["mandatory_completed"] += 1
                    self.current_job = None

            #  Job arrival or mode switch
            while self.events and self.events[0][0] <= self.current_time + 1e-9:

                # Event pop
                evt_time, evt_type, task_idx = heapq.heappop(self.events)

                if evt_type == 0:  # mode switch
                    if self.mode == 'LO': self._handle_mode_switch_to_hi()

                elif evt_type == 1:  # job arrival
                    task = self.tasks[task_idx]
                    next_arr = evt_time + task.period

                    # Event push
                    if next_arr < duration:
                        heapq.heappush(self.events, (next_arr, 1, task_idx))

                    self.stats["total_released"] += 1
                    job_id = self.job_seq_counters[task.id]
                    self.job_seq_counters[task.id] += 1

                    if task.criticality == "LO":
                        #if not getattr(task, 'is_backup_subblock', False):
                        #self.stats["lo_released_count"] += 1 # 可能有歧义，但是目前不用这个参数，暂时也不做约束
                        if not task.is_backup_subblock:
                            self.stats["weighted_released"] += task.utility
                        #self.stats["weighted_released"] += task.utility

                    # Determine WCET
                    if task.criticality == "HI" and self.mode == 'HI':
                        current_wcet = task.wcet_hi
                    elif task.is_backup_subblock:   #没啥影响，对于LO任务， wcet_hi=wcet_lo
                        # Backup always uses HI-like budget (stored in wcet_hi)
                        current_wcet = task.wcet_hi
                    else:
                        current_wcet = task.wcet_lo
                    # Set acet
                    acet = random.uniform(0.5 * current_wcet, current_wcet)
                    # Determine job type (mandatory or optional) and priority
                    is_optional = False
                    priority = task.priority
                    if task.criticality == "LO" and task.mk:
                        if self._get_pattern_bit(task, job_id) == 0:
                            is_optional = True
                            self.stats["optional_released"] += 1
                            priority = self.optional_prio_counter
                            self.optional_prio_counter += 1

                    new_job = Job(task_id=task.id, job_id=job_id, arrival_time=evt_time,
                                  deadline=evt_time + task.deadline,
                                  wcet=current_wcet, actual_exec_time=acet, priority=priority, is_optional=is_optional,
                                  criticality=task.criticality)

                    # [LOGGING] Job Arrival with ACET info
                    type_str = "OPTIONAL" if is_optional else "MANDATORY"
                    self.logger.info(
                        f"[{self.current_time:.2f}] [Arrival] Job T{task.id}-{job_id} ({type_str}) | WCET:{current_wcet:.2f} ACET:{acet:.2f}")
                    # --- Admission Control (Is the task added to the ready queue, 3 type jobs as follows:)---

                    reject_reason = ""
                    accept_reason = ""

                    accept = False
                    if task.criticality == "HI":
                        accept = True

                    # getattr(task, 'is_backup_subblock', False):

                    elif task.is_backup_subblock:
                        # 【备份子块】
                        # LO 模式: 拒绝 (休眠)
                        # HI 模式: 接受 (Mandatory 1, k)
                        if self.mode == 'HI' and not is_optional:
                            accept = True
                        # 备份块不跑 Optional

                    elif task.id in self.drop_list_ids:
                        # 【被 Drop 的原任务】
                        # LO 模式: 接受 (运行 m+x)
                        # HI 模式: 拒绝 (停止运行，交棒给 Backup)
                        if self.mode == 'LO':
                            # LO 模式下的 Optional 也要看 Slack
                            if not is_optional:
                                accept = True
                            elif self._calculate_interference_and_slack(new_job) >= new_job.initial_wcet - 1e-9:
                                accept = True
                        else:
                            accept = False  # HI 模式被 Drop

                    else:
                        # 【普通幸存 LO 任务】
                        # 始终接受 Mandatory
                        if not is_optional:
                            accept = True
                        elif self._calculate_interference_and_slack(new_job) >= new_job.initial_wcet - 1e-9:
                            accept = True

                    if accept:
                        self.ready_queue.append(new_job)
                        # [LOGGING] Admission Success
                        self.logger.info(f"    -> [ACCEPTED] Job Accept. Reason: {accept_reason}")
                    else:
                        # [LOGGING] Admission Rejection
                        self.logger.info(f"    -> [REJECTED] Job Dropped. Reason: {reject_reason}")

            # Dispatcher
            # Sort: Priority (lower number = higher priority), then arrival time
            self.ready_queue.sort(key=lambda j: (j.priority, j.arrival_time))

            if self.ready_queue:
                candidate = self.ready_queue[0]
                if self.current_job:
                    # Preemption check
                    if candidate.priority < self.current_job.priority:
                        # [LOGGING] Preemption
                        self.logger.info(
                            f"[{self.current_time:.2f}] [Preempt] T{candidate.task_id}-{candidate.job_id} preempts T{self.current_job.task_id}-{self.current_job.job_id}")
                        self.ready_queue.append(self.current_job)
                        self.current_job = candidate
                        self.ready_queue.pop(0)
                else:
                    # Processor idle
                    self.current_job = candidate
                    self.ready_queue.pop(0)
                    # [LOGGING] Start Execution
                    self.logger.info(
                        f"[{self.current_time:.2f}] [Dispatch] Job T{self.current_job.task_id}-{self.current_job.job_id} started.")

        # Count LO Released
        # lo_released_count = 0
        # for t in self.tasks:
        #     if t.criticality == "LO":
        #         lo_released_count += self.job_seq_counters[t.id]
        # self.stats['lo_released_count'] = lo_released_count
        # Count LO Executed
        #lo_executed_count = self.stats.get("lo_completed_count", 0)
        #self.stats['utility'] = lo_executed_count / lo_released_count if lo_executed_count > 0 else 0.0

        if self.stats["weighted_released"] > 0:
            self.stats['utility'] = self.stats["weighted_completed"] / self.stats["weighted_released"]
        else:
            self.stats['utility'] = 0.0

        # [LOGGING] Final Stats
        self.logger.info(f"--- Simulation End. Utility: {self.stats['utility']:.4f} ---")
        return self.stats


# def uaswc_online_simulation(tasks: List[Task], drop_list: List[Task], duration: int,
#                             mode_switch_time: float = float('inf')) -> Dict:
#     sim_tasks = copy.deepcopy(tasks)
#     sim = OnlineSimulator(sim_tasks, drop_list, mode_switch_time=mode_switch_time)
#     return sim.run(float(duration))


def run_multicore_simulation(processors: List[Processor],
                             duration: float,
                             mode_switch_time: float = float('inf')) -> Dict[str, Any]:
    """
    Runs the OnlineSimulator for each processor and aggregates the statistics.

    :param processors: List of Processor objects (with optimized tasks).
    :param duration: Simulation duration.
    :param mode_switch_time: Time to trigger mode switch (global assumption for simplicity).
    :return: Aggregated statistics dictionary.
    """
    #print(f"--- Running Multicore Simulation (Duration: {duration}, Switch: {mode_switch_time}) ---")

    # Initialize global stats accumulator
    global_stats = {
        "total_released": 0,
        "lo_released_count": 0,
        "lo_completed_count": 0,
        "mandatory_completed": 0,
        "optional_released": 0,
        "optional_completed": 0,
        "deadline_misses": 0,
        "weighted_released": 0.0,
        "weighted_completed": 0.0,
        "utility": 0.0
    }

    # Run simulation per core
    for p in processors:
        # Pass the task list and drop list of the specific processor
        sim = OnlineSimulator(p.tasks, p.drop_list, mode_switch_time=mode_switch_time)
        core_stats = sim.run(duration)

        # Aggregate results
        global_stats["total_released"] += core_stats["total_released"]
        global_stats["lo_released_count"] += core_stats["lo_released_count"]
        global_stats["lo_completed_count"] += core_stats["lo_completed_count"]
        global_stats["mandatory_completed"] += core_stats["mandatory_completed"]
        global_stats["optional_released"] += core_stats["optional_released"]
        global_stats["optional_completed"] += core_stats["optional_completed"]
        global_stats["deadline_misses"] += core_stats["deadline_misses"]
        global_stats["weighted_released"] += core_stats["weighted_released"]
        global_stats["weighted_completed"] += core_stats["weighted_completed"]

    if global_stats["weighted_released"] > 0:
        global_stats["utility"] = global_stats["weighted_completed"] / global_stats["weighted_released"]
    else:
        global_stats["utility"] = 0.0

    #print(f"--- Global Simulation Stats: Utility = {global_stats['utility']:.4f}, Misses = {global_stats['deadline_misses']} ---\n")
    return global_stats
