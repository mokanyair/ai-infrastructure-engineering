import csv
import json
import os
import platform
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_ID = os.environ["MODEL_ID"]
MODEL_REVISION = os.environ["MODEL_REVISION"]
CACHE_DIR = os.environ["HF_HOME"]

OUTPUT_DIR = Path(
    "/var/lib/ai-infra/benchmarks/phase9"
)

MAX_NEW_TOKENS = 64
WARMUP_RUNS = 1
MEASURED_RUNS = 3
SAMPLE_INTERVAL = 0.20

MESSAGES = [
    {
        "role": "user",
        "content": (
            "Explain in two short sentences why "
            "infrastructure engineers measure CPU "
            "utilization and memory usage during "
            "LLM inference."
        ),
    }
]

process = psutil.Process(os.getpid())
logical_cpus = psutil.cpu_count(logical=True)


def mib(value):
    return round(value / (1024 ** 2), 2)


def inference(model, tokenizer, inputs, attention_mask):
    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start

    generated = outputs[0, inputs.shape[1]:]
    output_tokens = generated.numel()

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    if not response or output_tokens <= 1:
        raise RuntimeError(
            "Inference returned invalid output."
        )

    return elapsed, output_tokens


def monitor_resources(stop_event, samples, run_number):
    # Prime counters so the first reported values
    # are not meaningless initialization readings.
    process.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)

    start = time.perf_counter()

    while not stop_event.is_set():
        process_cpu = process.cpu_percent(
            interval=SAMPLE_INTERVAL
        )

        host_cpu = psutil.cpu_percent(
            interval=None
        )

        memory = psutil.virtual_memory()

        samples.append({
            "run": run_number,
            "elapsed_seconds": round(
                time.perf_counter() - start,
                4,
            ),
            "process_cpu_percent_raw": round(
                process_cpu,
                2,
            ),
            "process_cpu_percent_normalized": round(
                process_cpu / logical_cpus,
                2,
            ),
            "host_cpu_percent": round(
                host_cpu,
                2,
            ),
            "process_rss_mib": mib(
                process.memory_info().rss
            ),
            "host_used_mib": mib(
                memory.used
            ),
            "host_available_mib": mib(
                memory.available
            ),
            "host_memory_percent": memory.percent,
        })


def main():
    torch.set_num_threads(4)

    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    print("Loading model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cpu")

    model.eval()

    inputs = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    attention_mask = torch.ones_like(inputs)

    print(
        f"Input tokens: {inputs.shape[1]}",
        flush=True,
    )

    print("Running one warm-up...", flush=True)

    inference(
        model,
        tokenizer,
        inputs,
        attention_mask,
    )

    all_samples = []
    run_results = []

    print(
        "Running measured resource-profile requests...",
        flush=True,
    )

    for run_number in range(1, MEASURED_RUNS + 1):
        samples = []
        stop_event = threading.Event()

        monitor = threading.Thread(
            target=monitor_resources,
            args=(
                stop_event,
                samples,
                run_number,
            ),
            daemon=True,
        )

        monitor.start()

        elapsed, output_tokens = inference(
            model,
            tokenizer,
            inputs,
            attention_mask,
        )

        stop_event.set()
        monitor.join()

        if not samples:
            raise RuntimeError(
                f"No resource samples for run {run_number}"
            )

        all_samples.extend(samples)

        raw_cpu = [
            x["process_cpu_percent_raw"]
            for x in samples
        ]

        normalized_cpu = [
            x["process_cpu_percent_normalized"]
            for x in samples
        ]

        host_cpu = [
            x["host_cpu_percent"]
            for x in samples
        ]

        rss = [
            x["process_rss_mib"]
            for x in samples
        ]

        result = {
            "run": run_number,
            "latency_seconds": round(
                elapsed,
                4,
            ),
            "output_tokens": output_tokens,
            "tokens_per_second": round(
                output_tokens / elapsed,
                4,
            ),
            "sample_count": len(samples),
            "mean_process_cpu_percent_raw": round(
                statistics.mean(raw_cpu),
                2,
            ),
            "peak_process_cpu_percent_raw": round(
                max(raw_cpu),
                2,
            ),
            "mean_process_cpu_percent_normalized": round(
                statistics.mean(normalized_cpu),
                2,
            ),
            "peak_process_cpu_percent_normalized": round(
                max(normalized_cpu),
                2,
            ),
            "mean_host_cpu_percent": round(
                statistics.mean(host_cpu),
                2,
            ),
            "peak_host_cpu_percent": round(
                max(host_cpu),
                2,
            ),
            "mean_process_rss_mib": round(
                statistics.mean(rss),
                2,
            ),
            "peak_process_rss_mib": round(
                max(rss),
                2,
            ),
        }

        run_results.append(result)

        print(
            f"Run {run_number}: "
            f"{result['latency_seconds']} s | "
            f"{result['tokens_per_second']} tok/s | "
            f"CPU normalized mean="
            f"{result['mean_process_cpu_percent_normalized']}% | "
            f"RSS peak="
            f"{result['peak_process_rss_mib']} MiB",
            flush=True,
        )

    summary = {
        "phase": "9.4",
        "measurement": "runtime_resource_profile",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "hostname": platform.node(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": "cpu",
        "dtype": "torch.float32",
        "logical_cpus": logical_cpus,
        "torch_threads": torch.get_num_threads(),
        "input_tokens": inputs.shape[1],
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "sample_interval_target_seconds":
            SAMPLE_INTERVAL,
        "mean_latency_seconds": round(
            statistics.mean(
                x["latency_seconds"]
                for x in run_results
            ),
            4,
        ),
        "mean_tokens_per_second": round(
            statistics.mean(
                x["tokens_per_second"]
                for x in run_results
            ),
            4,
        ),
        "mean_process_cpu_percent_normalized": round(
            statistics.mean(
                x[
                    "mean_process_cpu_percent_normalized"
                ]
                for x in run_results
            ),
            2,
        ),
        "peak_process_cpu_percent_normalized": round(
            max(
                x[
                    "peak_process_cpu_percent_normalized"
                ]
                for x in run_results
            ),
            2,
        ),
        "peak_process_rss_mib": round(
            max(
                x["peak_process_rss_mib"]
                for x in run_results
            ),
            2,
        ),
        "runs": run_results,
    }

    summary_path = (
        OUTPUT_DIR /
        "runtime-resource-profile.json"
    )

    samples_path = (
        OUTPUT_DIR /
        "runtime-resource-samples.csv"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with samples_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                all_samples[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(all_samples)

    print()
    print(
        "========== RESOURCE PROFILE SUMMARY =========="
    )
    print(json.dumps(summary, indent=2))

    print()
    print("Saved JSON:", summary_path)
    print("Saved samples:", samples_path)


if __name__ == "__main__":
    main()
