r"""
Static-(m,k) baseline.

No augmented levels in any mode:
    x_i^L = x_i^S = x_i^H = 0,   for all LO tasks.

No recovery in stable HI mode: Gamma_LO^rec = empty.

Serves as the lower-bound baseline.
"""

from typing import Dict, List

from core.processor import Processor
from core.task import Task
from experiments.performance import (
    global_max_importance,
    core_importance,
)


def run(processors: List[Processor],
        all_lo_tasks: List[Task],
        beta: float = 0.5) -> Dict[str, float]:
    """
    Apply Static-(m,k) to a partitioned system.

    Modifies processors in place (sets all x = 0, keeps drop_list as-is).

    Returns {mode: Perf^mode}.
    """
    for p in processors:
        for t in p.tasks:
            if t.criticality == "LO":
                t.mk.set_x(0, 'L')
                t.mk.set_x(0, 'S')
                t.mk.set_x(0, 'H')

    # Snapshot MC drop_list (same as final since no recovery)
    drops_mc = [list(p.drop_list) for p in processors]

    i_max = global_max_importance(all_lo_tasks, beta)
    if i_max == 0:
        return {'L': 0.0, 'S': 0.0, 'H': 0.0}

    total_l = sum(core_importance(p.tasks, [], beta, 'L') for p in processors)
    total_s = sum(core_importance(p.tasks, drops_mc[p.id], beta, 'S')
                  for p in processors)
    total_h = sum(core_importance(p.tasks, drops_mc[p.id], beta, 'H')
                  for p in processors)

    return {
        'L': total_l / i_max,
        'S': total_s / i_max,
        'H': total_h / i_max,
    }
