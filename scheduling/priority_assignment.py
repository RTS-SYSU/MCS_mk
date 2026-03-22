from typing import List

from core.task import Task


def assign_static_priorities(tasks: List[Task]) -> List[Task]:
    """
    Assign priorities to tasks based on:
    Use DMPO (Deadline Monotonic Priority Ordering: shorter deadline gets higher priority)

    Higher priority = smaller number.
    Result: task.priority field will be set.
    """

    # sort: lower value→ higher priority
    tasks_sorted = sorted(
        tasks,
        key=lambda t: (
            #0 if t.criticality == "HI" else 1,  # HI first
            t.deadline,  # DMPO: smaller relative deadline has higher priority
            t.period,  # tie-break: shorter period
            t.id  # tie-break: deterministic
        )
    )
    # Assign numeric priorities: 0 = highest priority
    for priority_value, task in enumerate(tasks_sorted):
        task.priority = priority_value

    return tasks_sorted
