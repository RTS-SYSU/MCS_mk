import math
from typing import List, Tuple
from core.task import Task
from scheduling.priority_assignment import assign_static_priorities


def schedulability_test(tasks: List[Task], drop_task=None) -> bool:
    """
    A schedulability test for MCS simulator with utility constraint based RTA.

    :param drop_task:
    :param tasks:a task set
    :return ture or false
    """
    # assign_static_priorities(tasks)

    if drop_task is None:
        drop_task = []
    for task in tasks:
        # Calculate LO mode WCRT
        R_lo, ok_lo = calculate_wcrt_lo(task, tasks)
        if not ok_lo:
            # print(R_lo, ok_lo)
            # print("cur", task)
            # for t in tasks:
            #     print(t)
            # breakpoint()
            # print(f"Task {task.id}: LO mode WCRT exceeded deadline or did not converge")

            return False
        # Calculate HI mode WCRT
        R_hi, ok_hi = calculate_wcrt_hi(task, tasks, drop_task)
        if not ok_hi:
            # print(f"Task {task.id}: HI mode WCRT exceeded deadline or did not converge")
            return False

        # Calculate mode change WCRT
        # calculate_wcrt_mc_terminate includes alculate_wcrt_mc if hpDL(i)==null
        R_mc, ok_mc = calculate_wcrt_mc_terminate(task, tasks, drop_task, R_lo)
        # R_mc, ok_mc = calculate_wcrt_mc(task, tasks, R_lo)
        if not ok_mc:
            # print(f"Task {task.id}: Mode change WCRT exceeded deadline")
            return False

    # All tasks passed the test
    return True


def schedulability_test_AMCrtbWH(tasks: List[Task]) -> bool:
    """
    A schedulability test for MCS simulator with utility constraint based RTA.
    cite:"Gettings O, Quinton S, Davis R I. Mixed criticality systems with weakly-hard constraints[C]
    //Proceedings of the 23rd International Conference on Real Time and Networks Systems. 2015: 237-246."

    :param tasks:a task set
    :return ture or false
    """
    for task in tasks:
        # Calculate LO-mode WCRT
        R_lo, ok_lo = calculate_wcrt_lo_AMCrtbWH(task, tasks)
        if not ok_lo:
            # print(f"Task {task.id}: LO-mode WCRT exceeded deadline or did not converge")

            return False

        # Calculate HI mode WCRT
        R_hi, ok_hi = calculate_wcrt_hi_AMCrtbWH(task, tasks)
        if not ok_hi:
            # print(f"Task {task.id}: HI mode WCRT exceeded deadline or did not converge")
            return False

        # Calculate mode change WCRT
        R_mc, ok_mc = calculate_wcrt_mc_AMCrtbWH(task, tasks, R_lo)
        if not ok_mc:
            # print(f"Task {task.id}: Mode change WCRT exceeded deadline")
            return False

    # All tasks passed the test
    return True


def test_aLO(task: Task, tasks: List[Task]) -> bool:
    """
    A schedulability test for a task in LO mode
    """
    R_lo, ok_lo = calculate_wcrt_lo(task, tasks)
    if not ok_lo:
        return False

    return True


def test_aHI(task: Task, tasks: List[Task], drop_task=None) -> bool:
    """
    A schedulability test for a task in HI mode
    """
    if drop_task is None:
        drop_task = []

    R_hi, ok_hi = calculate_wcrt_hi(task, tasks, drop_task)
    if not ok_hi:
        return False

    return True


def test_aMC(task: Task, tasks: List[Task], R_lo: float, drop_task=None) -> bool:
    """
    A schedulability test for a task in mode change
    """
    if drop_task is None:
        drop_task = []

    R_mc, ok_mc = calculate_wcrt_mc_terminate(task, tasks, drop_task, R_lo)
    if not ok_mc:
        return False

    return True


def calculate_wcrt_lo(cur_task: Task, tasks: List[Task], max_iter: int = 1000, tol: float = 1e-6) -> Tuple[float, bool]:
    """
    Calculate the worst-case response time (WCRT) of a task (LO or HI) in LO mode.

    Parameters:
        cur_task: The task to compute WCRT for.
        tasks: All tasks in the system, must have 'priority' assigned.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
    Returns:
        R_lo: WCRT in LO mode. Returns float('inf') if it does not converge or exceeds deadline.
        flag: True if WCRT converged and does not exceed deadline, False otherwise.
    """
    C_lo = cur_task.wcet_lo
    hp_tasks = [t for t in tasks if t.priority < cur_task.priority]
    hp_hi = [t for t in hp_tasks if t.criticality == "HI"]
    hp_lo = [t for t in hp_tasks if t.criticality == "LO"]

    R_prev = 0.0
    R_lo = C_lo
    iter_count = 0

    while abs(R_lo - R_prev) > tol and iter_count <= max_iter:
        R_prev = R_lo
        # higher priority HI tasks interference in LO mode
        interference_hi = sum(math.ceil(R_prev / t.period) * t.wcet_lo for t in hp_hi)

        # higher priority LO tasks interference in LO mode
        interference_lo = 0
        for task_q in hp_lo:
            if not task_q.is_backup_subblock:
                m_q = task_q.mk.m
                k_q = task_q.mk.k
                x_q = task_q.mk.x
                effective_m = m_q + x_q
                # d_q = math.ceil(R_prev / task_q.period) % k_q
                # if d_q == 0:
                #     num_lo = math.ceil(R_prev / (k_q * task_q.period)) * effective_m
                # else:
                #     num_lo = math.ceil(R_prev / (k_q * task_q.period)) * effective_m - max(effective_m - d_q, 0)
                num_lo = calculate_mk_jobs(R_prev, task_q.period, effective_m, k_q)
                interference_lo += num_lo * task_q.wcet_lo
        R_lo = C_lo + interference_hi + interference_lo
        # if R_lo < R_prev:
        #     print("Rlo",R_lo,"R_pre",R_prev)
        #     for t in tasks:
        #         print(t)
        #     print("curr",cur_task)
        #     print("count=",iter_count)
        #     breakpoint()
        if R_lo > cur_task.deadline:
            return R_lo, False
        iter_count += 1

    if iter_count > max_iter:
        flag = False
        # print(f"Warning: Task {cur_task.id} WCRT did not converge after {max_iter} iterations in LO mode")
    else:
        # flag indicates whether response time is within deadline
        flag = R_lo <= cur_task.deadline

    return R_lo, flag


def calculate_wcrt_hi(cur_task: Task, tasks: List[Task], drop_tasks: List[Task], max_iter: int = 1000,
                      tol: float = 1e-6) -> Tuple[float, bool]:
    """
    Calculate the worst-case response time (WCRT) of a task (LO or HI) in HI mode.

    Parameters:
        cur_task: The task to compute WCRT for.
        tasks: All tasks in the system, must have 'priority' assigned.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
    Returns:
        R_hi: WCRT in HI mode. Returns float('inf') if it does not converge or exceeds deadline.
        flag: True if WCRT converged and does not exceed deadline, False otherwise.
    """
    C_hi = cur_task.wcet_hi
    hp_tasks = [t for t in tasks if t.priority < cur_task.priority]
    hp_hi = [t for t in hp_tasks if t.criticality == "HI"]
    hp_lo = [t for t in hp_tasks if t.criticality == "LO" and t not in drop_tasks]

    R_prev = 0.0
    R_hi = C_hi
    iter_count = 0

    while abs(R_hi - R_prev) > tol and iter_count <= max_iter:
        R_prev = R_hi
        # higher priority HI tasks interference in HI mode
        interference_hi = sum(math.ceil(R_prev / t.period) * t.wcet_hi for t in hp_hi)

        # higher priority LO tasks interference in HI mode
        interference_lo = 0
        for task_q in hp_lo:
            m_q = task_q.mk.m
            k_q = task_q.mk.k
            # d_q = math.ceil(R_prev / task_q.period) % k_q
            # if d_q == 0:
            #     num_lo_hi = math.ceil(R_prev / (k_q * task_q.period)) * m_q
            # else:
            #     num_lo_hi = math.ceil(R_prev / (k_q * task_q.period)) * m_q - max(m_q - d_q, 0)
            num_lo_hi = calculate_mk_jobs(R_prev, task_q.period, m_q, k_q)
            interference_lo += num_lo_hi * task_q.wcet_lo
        R_hi = C_hi + interference_hi + interference_lo  # for LO task, Chi=Clo
        if R_hi > cur_task.deadline:
            return R_hi, False
        iter_count += 1

    if iter_count > max_iter:
        flag = False
        # print(f"Warning: Task {cur_task.id} WCRT did not converge after {max_iter} iterations in HI mode")
    else:
        # flag indicates whether response time is within deadline
        flag = R_hi <= cur_task.deadline

    return R_hi, flag


def calculate_wcrt_mc_terminate(cur_task: Task, tasks: List[Task], drop_tasks: List[Task], R_lo: float,
                                max_iter: int = 1000, tol: float = 1e-6) -> Tuple[
    float, bool]:
    """
    Calculate the worst-case response time (WCRT) of a task in mode-change (MC) and dropped some LO tasks in HI mode.

    Parameters:
        cur_task: Task to compute WCRT for.
        tasks: All tasks in the system (must have 'priority' assigned).
        R_lo: LO-mode WCRT of this task.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.

    Returns:
        R_mc: Mode-change WCRT (float)
        flag: True if R_mc <= deadline, False otherwise
    """

    if cur_task.criticality == "HI":
        C_max = cur_task.wcet_hi
    else:
        if cur_task in drop_tasks:
            return 0.0, True
        C_max = cur_task.wcet_lo

    # Identify higher priority tasks
    hp_tasks = [t for t in tasks if t.priority < cur_task.priority]
    hp_hi = [t for t in hp_tasks if t.criticality == "HI"]
    hp_lo = [t for t in hp_tasks if t.criticality == "LO"]

    R_prev = 0.0
    R_mc = R_lo  # initial guess
    iter_count = 0

    while abs(R_mc - R_prev) > tol and iter_count < max_iter:
        R_prev = R_mc

        # HI tasks interference in mode change (wcet_hi > wcet_lo)
        interference_hi = sum(math.ceil(R_prev / t.period) * t.wcet_hi for t in hp_hi)

        # LO tasks interference in mode change (LO + HI mode contributions combined)
        interference_lo_total = 0

        # for q in hp_lo:
        #     m_q = q.mk.m
        #     k_q = q.mk.k
        #     x_q = q.mk.x
        #     effective_m_q = m_q + x_q
        #
        #     if q in drop_tasks:
        #         count = calculate_mk_jobs(R_lo, q.period, effective_m_q, k_q)
        #         interference_lo_total += count * q.wcet_lo
        #     else:
        #         count_A = calculate_mk_jobs(R_lo, q.period, effective_m_q, k_q)
        #
        #         total_m = calculate_mk_jobs(R_prev, q.period, m_q, k_q)
        #         pre_m = calculate_mk_jobs(R_lo, q.period, m_q, k_q)
        #         count_B = max(0, total_m - pre_m)
        #
        #         interference_lo_total += (count_A + count_B) * q.wcet_lo
        for q in hp_lo:
            m_q = q.mk.m
            k_q = q.mk.k
            x_q = q.mk.x
            effective_m_q = m_q + x_q

            count = calculate_mk_jobs(R_lo, q.period, effective_m_q, k_q)
            interference_lo_total += count * q.wcet_lo

            if q not in drop_tasks:
                # count_A = calculate_mk_jobs(R_lo, q.period, effective_m_q, k_q)

                total_m = calculate_mk_jobs(R_prev, q.period, m_q, k_q)
                pre_m = calculate_mk_jobs(R_lo, q.period, m_q, k_q)
                count_B = max(0, total_m - pre_m)

                interference_lo_total += count_B * q.wcet_lo
        # Update R_mc
        R_mc = C_max + interference_hi + interference_lo_total
        if R_mc > cur_task.deadline:
            return R_mc, False
        iter_count += 1

    if iter_count >= max_iter:
        return R_mc, False
        # print(f"Warning: Task {cur_task.id} WCRT did not converge after {max_iter} iterations in mode-change")
    else:
        # flag indicates whether response time is within deadline
        return R_mc, (R_mc <= cur_task.deadline)


def calculate_mk_jobs(time_window: float, T: float, m: int, k: int) -> int:
    """Helper to calculate max jobs in a window given (m,k) constraint."""
    # Assuming Deeply-Red pattern for worst-case interference
    if time_window <= 0: return 0

    total_jobs = math.ceil(time_window / T)

    num_cycles = total_jobs // k
    remainder = total_jobs % k

    return num_cycles * m + min(remainder, m)


def calculate_wcrt_lo_AMCrtbWH(cur_task: Task, tasks: List[Task], max_iter: int = 1000, tol: float = 1e-6) -> Tuple[
    float, bool]:
    """
       Calculate the worst-case response time (WCRT) of a task (LO or HI) in LO mode.
       cite:"Gettings O, Quinton S, Davis R I. Mixed criticality systems with weakly-hard constraints[C]
       //Proceedings of the 23rd International Conference on Real Time and Networks Systems. 2015: 237-246."

       Parameters:
           cur_task: The task to compute WCRT for.
           tasks: All tasks in the system, must have 'priority' assigned.
           max_iter: Maximum number of iterations.
           tol: Convergence tolerance.
       Returns:
           R_lo: WCRT in LO mode. Returns float('inf') if it does not converge or exceeds deadline.
           flag: True if WCRT converged and does not exceed deadline, False otherwise.
       """
    C_lo = cur_task.wcet_lo
    hp_tasks = [t for t in tasks if t.priority < cur_task.priority]
    R_prev = 0.0
    R_lo = C_lo
    iter_count = 0

    while abs(R_lo - R_prev) > tol and iter_count <= max_iter:
        R_prev = R_lo
        # higher priority tasks interference in LO mode
        interference = sum(math.ceil(R_prev / t.period) * t.wcet_lo for t in hp_tasks)
        R_lo = C_lo + interference
        if R_lo > cur_task.deadline:
            return R_lo, False
        iter_count += 1

    if iter_count > max_iter:
        flag = False
        # print(f"Warning: Task {cur_task.id} WCRT did not converge after {max_iter} iterations in LO mode")
    else:
        # flag indicates whether response time is within deadline
        flag = R_lo <= cur_task.deadline

    return R_lo, flag


def calculate_wcrt_hi_AMCrtbWH(cur_task: Task, tasks: List[Task], max_iter: int = 1000, tol: float = 1e-6) -> Tuple[
    float, bool]:
    """
    Calculate the worst-case response time (WCRT) of a task (LO or HI) in HI mode.
    cite:"Gettings O, Quinton S, Davis R I. Mixed criticality systems with weakly-hard constraints[C]
    //Proceedings of the 23rd International Conference on Real Time and Networks Systems. 2015: 237-246."
    Parameters:
        cur_task: The task to compute WCRT for.
        tasks: All tasks in the system, must have 'priority' assigned.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
    Returns:
        R_hi: WCRT in HI mode. Returns float('inf') if it does not converge or exceeds deadline.
        flag: True if WCRT converged and does not exceed deadline, False otherwise.
    """
    C_hi = cur_task.wcet_hi
    hp_tasks = [t for t in tasks if t.priority < cur_task.priority]
    hp_hi = [t for t in hp_tasks if t.criticality == "HI"]
    hp_lo = [t for t in hp_tasks if t.criticality == "LO"]

    R_prev = 0.0
    R_hi = C_hi
    iter_count = 0

    while abs(R_hi - R_prev) > tol and iter_count <= max_iter:
        R_prev = R_hi
        # higher priority HI tasks interference in HI mode
        interference_hi = sum(math.ceil(R_prev / t.period) * t.wcet_hi for t in hp_hi)

        # higher priority LO tasks interference in HI mode
        interference_lo = 0
        for task_q in hp_lo:
            T_q = task_q.period
            m_q = task_q.mk.m
            k_q = task_q.mk.k
            s_q = k_q - m_q
            num_lo_hi = math.ceil(R_prev / T_q)

            subtract_num = 0
            for n in range(1, s_q + 1):
                subtract_num += math.ceil((R_prev - (k_q - n) * T_q) / (T_q * k_q))

            interference_lo += (num_lo_hi - subtract_num) * task_q.wcet_lo

        R_hi = C_hi + interference_hi + interference_lo
        if R_hi > cur_task.deadline:
            return R_hi, False
        iter_count += 1

    if iter_count > max_iter:
        flag = False
        # print(f"Warning: Task {cur_task.id} WCRT did not converge after {max_iter} iterations in HI mode")
    else:
        # flag indicates whether response time is within deadline
        flag = R_hi <= cur_task.deadline

    return R_hi, flag


def calculate_wcrt_mc_AMCrtbWH(cur_task: Task, tasks: List[Task], R_lo: float, max_iter: int = 1000,
                               tol: float = 1e-6) -> Tuple[float, bool]:
    """
    Calculate the worst-case response time (WCRT) of a task (LO or HI) in mode change.
    cite:"Gettings O, Quinton S, Davis R I. Mixed criticality systems with weakly-hard constraints[C]
    //Proceedings of the 23rd International Conference on Real Time and Networks Systems. 2015: 237-246."
    Parameters:
        cur_task: The task to compute WCRT for.
        tasks: All tasks in the system, must have 'priority' assigned.
        R_lo: LO-mode WCRT of this task.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
    Returns:
        R_hi: WCRT in HI mode. Returns float('inf') if it does not converge or exceeds deadline.
        flag: True if WCRT converged and does not exceed deadline, False otherwise.
    """
    hp_tasks = [t for t in tasks if t.priority < cur_task.priority]
    hp_hi = [t for t in hp_tasks if t.criticality == "HI"]
    hp_lo = [t for t in hp_tasks if t.criticality == "LO"]

    R_prev = 0.0
    R_mc = R_lo  # initial guess
    iter_count = 0
    if cur_task.criticality == "HI":
        while abs(R_mc - R_prev) > tol and iter_count <= max_iter:
            R_prev = R_mc

            # HI tasks interference in mode change (wcet_hi > wcet_lo)
            interference_hi = sum(math.ceil(R_prev / t.period) * t.wcet_hi for t in hp_hi)

            # LO tasks interference in mode change
            interference_lo = 0
            for task_q in hp_lo:
                T_q = task_q.period
                m_q = task_q.mk.m
                k_q = task_q.mk.k
                s_q = k_q - m_q
                num_lo_hi = math.ceil(R_prev / T_q)
                x = math.ceil(R_lo / T_q) * T_q
                subtract_num = 0
                for n in range(s_q, k_q + 1):
                    subtract_num += max(math.ceil((R_prev - (k_q - n) * T_q - x) / (T_q * k_q)), 0)

                interference_lo += (num_lo_hi - subtract_num) * task_q.wcet_lo

            R_mc = cur_task.wcet_hi + interference_hi + interference_lo
            if R_mc > cur_task.deadline:
                return R_mc, False
            iter_count += 1
    else:
        while abs(R_mc - R_prev) > tol and iter_count <= max_iter:
            R_prev = R_mc
            # HI tasks interference in mode change (wcet_hi > wcet_lo)
            interference_hi = sum(math.ceil(R_prev / t.period) * t.wcet_hi for t in hp_hi)

            # LO tasks interference in mode change
            interference_lo = sum(math.ceil(R_prev / t.period) * t.wcet_lo for t in hp_lo)

            R_mc = cur_task.wcet_lo + interference_hi + interference_lo
            if R_mc > cur_task.deadline:
                return R_mc, False
            iter_count += 1

    if iter_count > max_iter:
        flag = False
        # print(f"Warning: Task {cur_task.id} WCRT did not converge after {max_iter} iterations in HI mode")
    else:
        # flag indicates whether response time is within deadline
        flag = R_mc <= cur_task.deadline

    return R_mc, flag
