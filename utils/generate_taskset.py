import random
from typing import List, Tuple
import numpy as np

from core.task import Task
from utils.drs import drs


def generate_task_periods(n: int, max_hyperperiod: int = 1440) -> List[int]:
    """
    Generate `n` task periods such that the system hyperperiod (LCM of all periods)
    does not exceed `max_hyperperiod`. All generated task periods are therefore
    divisors of `max_hyperperiod`.

    Parameters:
        n (int): Number of tasks.
        max_hyperperiod (int): Upper bound of the hyperperiod (default: 1440).
    Returns:
        List[int]: A list containing `n` task periods.
    """

    divisors = [i for i in range(1, max_hyperperiod + 1) if max_hyperperiod % i == 0]
    return [random.choice(divisors) for _ in range(n)]

def generate_task_periods_2(n: int) -> List[int]:
    """
    [MODIFIED] Generate n task periods.
    New logic: Randomly select integers from [1, 500].
    The 'max_hyperperiod' argument is ignored in this logic but kept for interface compatibility.
    """
    # 周期必须 >= 1，0是非法的
    return [random.randint(1, 100) for _ in range(n)]

def generate_task_utilizations(total_processor: int, total_task: int, targetU: float,
                               cp: float, cf: float, xf: float) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate task utilizations for mixed-criticality system based on DRS.

    Parameters:
        total_processor (float): Total processing capacity (sum of CPU).
        total_task (int): Total number of tasks.
        targetU (float): Target total utilization (sum of all tasks).
        cp (float): Fraction of HI tasks, i.e., the number of HI tasks is equal to cp*totalTask.
        cf (float): Ratio for HI tasks: C_HI = cf * C_LO
        xf (float): Ratio for LO tasks: C_HI = xf * C_LO

   Returns:
        Tuple[List[List[float]], List[List[float]]]:
            (LO_tasks, HI_tasks), each task is [C_LO, C_HI]
    """
    n_hi = int(total_task * cp)
    n_lo = total_task - n_hi

    # Total utilization for LO tasks
    ULO = total_processor * targetU
    U_lo_lo = ULO * (1 - cp)  # utilization: LO tasks in LO mode
    U_hi_lo = ULO - U_lo_lo  # utilization: HI tasks in LO mode
    U_lo_hi = xf * U_lo_lo  # utilization: LO tasks in HI mode
    U_hi_hi = cf * U_hi_lo  # utilization: HI tasks in HI mode

    # Generate HI task utilizations
    vec_U_hi_hi_upbound = np.ones(n_hi)
    vec_U_hi_hi = drs(n_hi, U_hi_hi, vec_U_hi_hi_upbound)  # HI tasks in HI mode
    vec_U_hi_lo = drs(n_hi, U_hi_lo, vec_U_hi_hi)  # HI tasks in LO mode

    # Generate LO task utilizations
    vec_U_lo_lo_upBound = np.ones(n_lo)
    vec_U_lo_lo = drs(n_lo, U_lo_lo, vec_U_lo_lo_upBound)  # LO tasks in LO mode
    vec_U_lo_hi = drs(n_lo, U_lo_hi, vec_U_lo_lo)  # LO tasks in HI mode

    # Combine results per task: [C_LO, C_HI]

    # HI tasks first
    hi_task_utils = [[vec_U_hi_lo[i], vec_U_hi_hi[i]] for i in range(n_hi)]
    # LO tasks
    lo_task_utils = [[vec_U_lo_lo[i], vec_U_lo_hi[i]] for i in range(n_lo)]

    return hi_task_utils, lo_task_utils


def generate_taskset(total_processor: int = 1.0,
                     total_task: int = 10,
                     targetU: float = 0.6,
                     cp: float = 0.5,
                     cf: float = 2.0,
                     xf: float = 1.0,
                     max_hyperperiod: int = 1440,
                     m: int = None,
                     k: int = None) -> List[Task]:
    """
    Generate a full mixed-criticality task set.
    If m and k are not forcibly specified externally, they will be randomly generated
     Parameters:
        - total_processor (float): Total processing capacity (sum of CPU).
        - total_task (int): Total number of tasks.
        - targetU (float): Target total utilization (sum of all tasks).
        - cp (float): Fraction of HI tasks, i.e., the number of HI tasks is equal to cp*totalTask.
        - cf (float): Ratio for HI tasks: C_HI = cf * C_LO
        - xf (float): Ratio for LO tasks: C_HI = xf * C_LO
        - max_hyperperiod (int): least common multiple(lcm) of periods
        - m: (m,k)-constraint
        - k: (k,m)-constraint
     Returns:
        - tasks: List[Task]
    """

    if m is not None and k is None:
        raise ValueError("Invalid arguments: 'k' must be provided if 'm' is specified.")
    if k is not None and m is None:
        raise ValueError("Invalid arguments: 'm' must be provided if 'k' is specified.")

    if m is not None and k is not None and m >k:
        raise ValueError(f"Invalid (m,k) constraint: m={m} must be less than or equal to k={k}.")
    if (m is not None and m <= 0) or (k is not None and k <= 0):
        raise ValueError(f"Invalid arguments: m and k must be > 0. Got m={m}, k={k}")

    #periods = generate_task_periods(total_task, max_hyperperiod)
    periods = generate_task_periods(total_task,max_hyperperiod)
    hi_utils, lo_utils = generate_task_utilizations(total_processor, total_task, targetU, cp, cf, xf)

    tasks: List[Task] = []
    n_hi = int(total_task * cp)
    n_lo = total_task - n_hi

    # Create HI tasks
    for i in range(n_hi):
        tasks.append(Task(id=i + 1, criticality="HI",
                          period=periods[i], deadline=periods[i],
                          wcet_lo=hi_utils[i][0] * periods[i], wcet_hi=hi_utils[i][1] * periods[i]))

    # Create LO tasks
    for i in range(n_lo):
        # If m and k are not forcibly specified externally, they will be randomly generated
        this_k = k if k is not None else random.randint(2, 10)
        this_m = m if m is not None else random.randint(1, this_k - 1)
        idx = n_hi + i

        utility = random.random()*9+1
        #utility = random.choice([1, 100])
        #utility = random.uniform(1, 100)
        # u_val = lo_utils[i][0]
        # if u_val < 0.02:
        #     utility = 100
        # else:
        #     utility = 1
        tasks.append(Task(id=idx + 1, criticality="LO",
                          period=periods[idx], deadline=periods[idx],
                          wcet_lo=lo_utils[i][0] * periods[idx], wcet_hi=lo_utils[i][1] * periods[idx], m=this_m,
                          k=this_k,utility=utility))

    return tasks


# if __name__ == "__main__":
#
#     sum1 = 0
#     sum2 = 0
#     for i in range(1):
#         tasks = generate_taskset(1, 10, 0.6, 0.5, 2, 1, 1440)
#
#         for task in tasks:
#             print(task)
#         assign_static_priorities(tasks)
#         # for task in tasks:
#         #     print(task)
#         if schedulability_test(tasks):
#             sum1 += 1
#
#         if schedulability_test_AMCrtbWH(tasks):
#             sum2 += 1
#
#     print(sum1, sum2)
