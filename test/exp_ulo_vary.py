import copy
import logging
import pandas as pd
import multiprocessing
from typing import Dict, Any, Tuple, Callable, Optional

from scheduling.normalization_utility import calculate_hyperperiod, get_normalization_utility_offline_stats
from scheduling.offline_simulator import uaswc_offline_multicore, cost_utility_density
from scheduling.online_simulator import run_multicore_simulation
from scheduling.sched_test import schedulability_test
from scheduling.task_partitioning import partition_only
from scheduling.priority_assignment import assign_static_priorities

from utils.generate_taskset import generate_taskset



def _worker_sample(args: Dict[str, Any]) -> Dict[str, float]:
    """
    Single sample worker. Returns utilities for three methods for one generated taskset.
    Method1: fixed (random m,k) - no optimization. If schedulable -> avg(m/k) over LO tasks, else 0.
    Method2: fixed (m=k=1) - no optimization. If schedulable -> avg(m/k) (==1) else 0.
    Method3: UASWC offline (cost_utility_density). If offline reports feasible -> normalized LO utility, else 0.
    """
    gen_params = args['gen_params']
    num_cores = args['num_cores']

    # 1. generate base taskset (we'll clone for different methods)
    try:
        tasks_base = generate_taskset(**gen_params)
    except ValueError:
        # invalid parameters; treat as all-zero utilities
        return {"M1": 0.0, "M2": 0.0, "M3": 0.0, "M1_ok": 0, "M2_ok": 0, "M3_ok": 0}

    # prepare results
    # track LO/HI normalized offline utilities for each method and online utility for M3
    res = {"M1_lo": 0.0, "M1_hi": 0.0, "M2_lo": 0.0, "M2_hi": 0.0,
        "M3_lo": 0.0, "M3_hi": 0.0, "M3_on": 0.0, "M3_on_hi": 0.0,
        "M1_ok": 0, "M2_ok": 0, "M3_ok": 0}

    # compute global hyperperiod and base_total_value for normalization (used in method3)
    global_hyperperiod = calculate_hyperperiod(tasks_base)
    if global_hyperperiod == 0:
        return res

    _, base_total_value = get_normalization_utility_offline_stats(tasks_base, [], mode='LO', grid_hyperperiod=global_hyperperiod)
    if base_total_value == 0:
        # nothing to normalize against
        return res

    # ------------------ Method 1: fixed random (m,k) ------------------
    tasks_m1 = copy.deepcopy(tasks_base)
    processors_m1 = partition_only(tasks_m1, num_cores)
    if processors_m1 is not None:
        ok = True
        for p in processors_m1:
            assign_static_priorities(p.tasks)
            if not schedulability_test(p.tasks):
                ok = False
                break
        if ok:
            lo_executed_sum = 0.0
            hi_executed_sum = 0.0
            base_lo_total = 0.0
            base_hi_total = 0.0
            for p in processors_m1:
                e_lo, total_lo = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='LO', grid_hyperperiod=global_hyperperiod)
                e_hi, total_hi = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='HI', grid_hyperperiod=global_hyperperiod)
                lo_executed_sum += e_lo
                hi_executed_sum += e_hi
                base_lo_total += total_lo
                base_hi_total += total_hi

            res['M1_lo'] = (lo_executed_sum / base_lo_total) if base_lo_total > 0 else 0.0
            res['M1_hi'] = (hi_executed_sum / base_hi_total) if base_hi_total > 0 else 0.0
            res['M1_ok'] = 1

    # ------------------ Method 2: fixed m=k (modify copy, do NOT regenerate) ------------------
    tasks_m2 = copy.deepcopy(tasks_base)
    # force m = k for all LO tasks (i.e., all jobs mandatory)
    for t in tasks_m2:
        if t.criticality == 'LO' and t.mk:
            t.mk.m = 1
            t.mk.k = 1
            # refresh patterns
            try:
                t.mk.update_pattern(is_degraded=False)
                t.mk.update_pattern(is_degraded=True)
            except Exception:
                pass

    processors_m2 = partition_only(tasks_m2, num_cores)
    if processors_m2 is not None:
        ok2 = True
        for p in processors_m2:
            assign_static_priorities(p.tasks)
            if not schedulability_test(p.tasks):
                ok2 = False
                break
        if ok2:
            lo_executed_sum = 0.0
            hi_executed_sum = 0.0
            base_lo_total = 0.0
            base_hi_total = 0.0
            for p in processors_m2:
                e_lo, total_lo = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='LO', grid_hyperperiod=global_hyperperiod)
                e_hi, total_hi = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='HI', grid_hyperperiod=global_hyperperiod)
                lo_executed_sum += e_lo
                hi_executed_sum += e_hi
                base_lo_total += total_lo
                base_hi_total += total_hi

            res['M2_lo'] = (lo_executed_sum / base_lo_total) if base_lo_total > 0 else 0.0
            res['M2_hi'] = (hi_executed_sum / base_hi_total) if base_hi_total > 0 else 0.0
            res['M2_ok'] = 1

    # ------------------ Method 3: UASWC offline ------------------
    tasks_m3 = copy.deepcopy(tasks_base)
    # run offline optimizer
    is_feasible, processors = uaswc_offline_multicore(original_tasks=tasks_m3, num_processors=num_cores,
                                                     cost_func=cost_utility_density, is_dynamic_strategy=False)
    if is_feasible:
        lo_executed_sum = 0.0
        hi_executed_sum = 0.0
        for p in processors:
            e_lo, _ = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='LO', grid_hyperperiod=global_hyperperiod)
            e_hi, _ = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='HI', grid_hyperperiod=global_hyperperiod)
            lo_executed_sum += e_lo
            hi_executed_sum += e_hi
        res['M3_lo'] = (lo_executed_sum / base_total_value) if base_total_value > 0 else 0.0
        _, base_total_hi = get_normalization_utility_offline_stats(tasks_base, [], mode='HI', grid_hyperperiod=global_hyperperiod)
        res['M3_hi'] = (hi_executed_sum / base_total_hi) if base_total_hi > 0 else 0.0
        res['M3_ok'] = 1

    # --- Online evaluation: run the multicore online simulator in stable LO mode ---
    sim_stats = run_multicore_simulation(processors, duration=global_hyperperiod, mode_switch_time=float('inf'))
    # sim_stats returns {'utility': ...}
    res['M3_on'] = sim_stats.get('utility', 0.0)

    # --- Online evaluation: run the multicore online simulator in stable HI mode ---
    sim_stats_hi = run_multicore_simulation(processors, duration=global_hyperperiod, mode_switch_time=0.0)
    res['M3_on_hi'] = sim_stats_hi.get('utility', 0.0)

    return res


def run_single_util_point(target_util: float,
                          num_cores: int,
                          fixed_params: Dict[str, Any],
                          test_times: int) -> Tuple[Dict[str, float], Dict[str, int]]:
    current_gen_params = fixed_params.copy()
    current_gen_params['targetU'] = target_util
    current_gen_params['total_processor'] = num_cores

    batch_args = [{'gen_params': current_gen_params, 'num_cores': num_cores} for _ in range(test_times)]

    num_processes = 4
    with multiprocessing.Pool(processes=min(num_processes, test_times)) as pool:
        results = pool.map(_worker_sample, batch_args)

    # aggregate
    agg = {'M1_lo': 0.0, 'M1_hi': 0.0, 'M2_lo': 0.0, 'M2_hi': 0.0, 'M3_lo': 0.0, 'M3_hi': 0.0, 'M3_on': 0.0, 'M3_on_hi': 0.0}
    counts = {'M1_ok': 0, 'M2_ok': 0, 'M3_ok': 0}

    for r in results:
        agg['M1_lo'] += r.get('M1_lo', 0.0)
        agg['M1_hi'] += r.get('M1_hi', 0.0)
        agg['M2_lo'] += r.get('M2_lo', 0.0)
        agg['M2_hi'] += r.get('M2_hi', 0.0)
        agg['M3_lo'] += r.get('M3_lo', 0.0)
        agg['M3_hi'] += r.get('M3_hi', 0.0)
        agg['M3_on'] += r.get('M3_on', 0.0)
        agg['M3_on_hi'] += r.get('M3_on_hi', 0.0)
        counts['M1_ok'] += r.get('M1_ok', 0)
        counts['M2_ok'] += r.get('M2_ok', 0)
        counts['M3_ok'] += r.get('M3_ok', 0)

    # average over total samples (including zeros for failed ones as requested)
    for k in list(agg.keys()):
        agg[k] /= test_times

    return agg, counts


def main():
    NUM_CORES = 4
    TEST_TIMES = 100  # small default for smoke test; increase as needed

    FIXED_PARAMS = {
        'total_task': 20 * NUM_CORES,
        'cp': 0.5, 'cf': 2, 'xf': 1.0,
        'max_hyperperiod': 1440,
        'm': None, 'k': None
    }

    u_values = [x * 0.05 for x in range(13, 19)]
    final = []

    print(f"Starting ULO-varying experiment | cores={NUM_CORES} | samples per point={TEST_TIMES}")
    for u in u_values:
        agg, counts = run_single_util_point(u, NUM_CORES, FIXED_PARAMS, TEST_TIMES)
        row = {"Utilization": u,
      'M1_lo_norm': agg['M1_lo'], 'M1_hi_norm': agg['M1_hi'], 'M1_feasible': counts['M1_ok'],
      'M2_lo_norm': agg['M2_lo'], 'M2_hi_norm': agg['M2_hi'], 'M2_feasible': counts['M2_ok'],
      'M3_lo_norm': agg['M3_lo'], 'M3_hi_norm': agg['M3_hi'], 'M3_feasible': counts['M3_ok'],
      'M3_online_lo': agg['M3_on'], 'M3_online_hi': agg['M3_on_hi']}
        final.append(row)
        print(f"U={u:.2f} | M1_LO={row['M1_lo_norm']:.3f} M1_HI={row['M1_hi_norm']:.3f} (ok {row['M1_feasible']}) | M2_LO={row['M2_lo_norm']:.3f} M2_HI={row['M2_hi_norm']:.3f} (ok {row['M2_feasible']}) | M3_LO={row['M3_lo_norm']:.3f} M3_HI={row['M3_hi_norm']:.3f} (ok {row['M3_feasible']}) | M3_on_lo={row['M3_online_lo']:.3f} M3_on_hi={row['M3_online_hi']:.3f}")

    df = pd.DataFrame(final)
    filename = f"section1_result_ulo_vary_{NUM_CORES}core"
    df.to_excel(f"{filename}_full.xlsx", index=False)
    print(f"Saved results to {filename}_full.xlsx")


if __name__ == '__main__':
    main()
