# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU core affinity utilities for heterogeneous ARM big.LITTLE systems."""

from __future__ import annotations

import os
import platform

from vllm.logger import init_logger

logger = init_logger(__name__)


def pin_to_performance_cores() -> None:
    """Pin the current thread to performance CPU cores on ARM big.LITTLE.

    On heterogeneous ARM SoCs (e.g. NVIDIA GB10 with Cortex-X925 + A725),
    the Linux scheduler may migrate compute-heavy threads between fast and
    slow cores, causing bimodal latency. Pinning to the fastest cores
    eliminates this variance.

    Falls back silently on x86 or when core speeds can't be determined.
    """
    if platform.machine() not in ("aarch64", "arm64"):
        return

    try:
        max_freqs: dict[int, int] = {}
        for cpu_id in os.sched_getaffinity(0):
            freq_path = (
                f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/cpuinfo_max_freq"
            )
            try:
                with open(freq_path) as f:
                    max_freqs[cpu_id] = int(f.read().strip())
            except (FileNotFoundError, ValueError):
                continue

        if not max_freqs:
            return

        top_freq = max(max_freqs.values())
        perf_cores = {
            cpu for cpu, freq in max_freqs.items() if freq == top_freq
        }

        if len(perf_cores) < len(max_freqs):
            os.sched_setaffinity(0, perf_cores)
            logger.info(
                "Pinned engine thread to %d performance cores: %s"
                " (max %d MHz)",
                len(perf_cores),
                sorted(perf_cores),
                top_freq // 1000,
            )
    except Exception as e:
        logger.debug("Could not pin to performance cores: %s", e)
