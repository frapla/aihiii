import logging
import multiprocessing as mp
import sys
import time
from itertools import product, islice
from pathlib import Path
from time import perf_counter
from typing import List, Union, Tuple

import numpy as np
import psutil

sys.path.append(str(Path(__file__).absolute().parents[2]))
from src.utils.iso18571 import ISO18571
from src.utils.iso18571_openvt import ISO18571 as ISO18571Openvt

LOG: logging.Logger = logging.getLogger(__name__)


def batched(iterable, n):
    while batch := tuple(islice(iterable, n)):
        yield batch


def performance_watcher(func):
    def wrapper(*args, **kwargs):
        LOG.info("Running %s", func.__name__)
        at_main, at_sub = mp.Pipe()
        proc = mp.Process(target=memory_whatch, args=[at_sub])
        proc.start()

        elapsed = perf_counter()
        ret = func(*args, **kwargs)
        elapsed = perf_counter() - elapsed

        at_main.send(True)
        proc.join()
        memory = at_main.recv()
        at_main.close()
        at_sub.close()
        proc.close()

        LOG.info("Finished %s with %s comparisons in %.4fs - Max RAM usage %.2f", func.__name__, len(ret), elapsed, memory)

        return elapsed, memory

    return wrapper


def memory_whatch(con):
    mem = 0
    while not con.poll():
        if sys.platform == "linux":
            mem_ = psutil.virtual_memory().active
        else:
            mem_ = psutil.virtual_memory().used
        mem = max([mem, mem_ * 1e-9])
        time.sleep(0.01)
    else:
        con.send(mem)


def test():
    at_main, at_sub = mp.Pipe()
    proc = mp.Process(target=memory_whatch, args=[at_sub])

    proc.start()
    time.sleep(1)
    at_main.send(True)
    proc.join()
    print("got", at_main.recv())
    at_main.close()
    at_sub.close()
    proc.close()


def get_iso_own(signals: List[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
    results = []
    iso = ISO18571()
    for signal_ref, signal_comp in signals:
        results.append(
            iso.rating_iso_18571(
                signal_ref=signal_ref,
                signal_comp=signal_comp,
            )["ISO 18571 Rating"]
        )

    return results


def get_iso_openvt(signal_ref: np.ndarray, signal_comp: np.ndarray) -> float:
    iso = ISO18571Openvt(
        reference_curve=np.stack([np.zeros(signal_ref.shape[0]), signal_ref]).T,
        comparison_curve=np.stack([np.zeros(signal_comp.shape[0]), signal_comp]).T,
    )

    return iso.overall_rating()


@performance_watcher
def run_sequential(
    function: Union[get_iso_openvt, get_iso_own], signals_ref: List[np.ndarray], signals_comp: List[np.ndarray]
) -> float:
    result = []
    for signal_ref, signal_comp in product(signals_ref, signals_comp):
        result.append(function(signal_ref, signal_comp))

    return result


@performance_watcher
def run_mp_pool(
    function: Union[get_iso_openvt, get_iso_own],
    signals_ref: List[np.ndarray],
    signals_comp: List[np.ndarray],
    cpu_count: int = mp.cpu_count(),
    chunk_size: int = 1,
) -> float:
    with mp.Pool(processes=cpu_count) as pool:
        result = pool.map(function, batched(product(signals_ref, signals_comp), chunk_size))

    return result


if __name__ == "__main__":
    test()
