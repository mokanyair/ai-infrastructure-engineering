import csv
import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

MODEL_REVISION = os.environ["MODEL_REVISION"]
CACHE_DIR = os.environ["HF_HOME"]

OUTPUT_DIR = Path(
    "/var/lib/ai-infra/benchmarks/phase9"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_INTERVAL = 0.05

process = psutil.Process(os.getpid())

samples = []
stop_event = threading.Event()


def mib(value):
    return round(value / (1024 ** 2), 2)


def sample_memory():
    start = time.perf_counter()

    while not stop_event.is_set():
        try:
            rss = process.memory_info().rss
            host = psutil.virtual_memory()

            samples.append({
                "elapsed_seconds": round(
                    time.perf_counter() - start, 4
                ),
                "process_rss_mib": mib(rss),
                "host_used_mib": mib(host.used),
                "host_available_mib": mib(
                    host.available
                ),
                "host_memory_percent": host.percent,
            })

        except psutil.NoSuchProcess:
            break

        stop_event.wait(SAMPLE_INTERVAL)


def main():
    torch.set_num_threads(4)

    rss_before = mib(
        process.memory_info().rss
    )

    monitor = threading.Thread(
        target=sample_memory,
        daemon=True,
    )

    print("Starting memory monitoring...", flush=True)

    monitor.start()

    model = None
    load_start = time.perf_counter()

    try:
        print("Loading pinned model...", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=CACHE_DIR,
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        model = model.to("cpu")
        model.eval()

        load_seconds = (
            time.perf_counter() - load_start
        )

        rss_after = mib(
            process.memory_info().rss
        )

    finally:
        stop_event.set()
        monitor.join()

    if not samples:
        raise RuntimeError(
            "No memory samples were collected."
        )

    peak_rss = max(
        sample["process_rss_mib"]
        for sample in samples
    )

    result = {
        "phase": "9.2",
        "measurement": "model_initialization",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "hostname": platform.node(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": "cpu",
        "dtype": "torch.float32",
        "cpu_threads": torch.get_num_threads(),
        "sample_interval_target_seconds":
            SAMPLE_INTERVAL,
        "sample_count": len(samples),
        "model_load_seconds": round(
            load_seconds, 4
        ),
        "rss_before_model_mib": rss_before,
        "rss_after_model_mib": rss_after,
        "observed_peak_rss_mib": peak_rss,
        "observed_peak_rss_delta_mib": round(
            peak_rss - rss_before, 2
        ),
        "host_available_after_mib": mib(
            psutil.virtual_memory().available
        ),
    }

    json_path = OUTPUT_DIR / (
        "model-load-profile.json"
    )

    csv_path = OUTPUT_DIR / (
        "model-load-samples.csv"
    )

    json_path.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(samples[0].keys()),
        )

        writer.writeheader()
        writer.writerows(samples)

    print()
    print("========== MODEL LOAD PROFILE ==========")
    print(json.dumps(result, indent=2))

    print()
    print("Saved JSON:", json_path)
    print("Saved CSV:", csv_path)

    # Keep the model alive until measurements
    # and evidence have been written.
    del model


if __name__ == "__main__":
    main()
