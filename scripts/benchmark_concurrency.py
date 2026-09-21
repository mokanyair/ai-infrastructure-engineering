import csv
import json
import os
import platform
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ["MODEL_ID"]
REVISION = os.environ["MODEL_REVISION"]
CACHE_DIR = os.environ["HF_HOME"]

OUTPUT_DIR = Path(
    "/var/lib/ai-infra/benchmarks/phase9"
)

THREADS = 4
MAX_NEW_TOKENS = 64
CONCURRENCY_LEVELS = [1, 2, 4]
SAMPLE_INTERVAL = 0.2

PROMPT = (
    "Explain in two short sentences why infrastructure "
    "engineers measure CPU utilization and memory usage "
    "during LLM inference."
)

process = psutil.Process(os.getpid())
logical_cpus = psutil.cpu_count(logical=True)


def mib(value):
    return round(value / (1024 ** 2), 2)


def monitor_resources(stop_event, samples, concurrency):
    process.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)

    start = time.perf_counter()

    while not stop_event.is_set():
        process_cpu = process.cpu_percent(
            interval=SAMPLE_INTERVAL
        )

        host_cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        samples.append({
            "concurrency": concurrency,
            "elapsed_seconds": round(
                time.perf_counter() - start,
                4,
            ),
            "process_cpu_raw_percent": round(
                process_cpu,
                2,
            ),
            "process_cpu_normalized_percent": round(
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
            "host_available_mib": mib(
                memory.available
            ),
        })


def run_request(
    request_id,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    start_barrier,
):
    start_barrier.wait()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start

    generated = output[
        0,
        input_ids.shape[1]:,
    ]

    output_tokens = generated.numel()

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    if not response or output_tokens <= 1:
        raise RuntimeError(
            f"Request {request_id} returned invalid output."
        )

    return {
        "request_id": request_id,
        "output_tokens": int(output_tokens),
        "latency_seconds": round(
            elapsed,
            4,
        ),
        "effective_output_tokens_per_second": round(
            output_tokens / elapsed,
            4,
        ),
        "success": True,
    }


def main():
    torch.set_num_threads(THREADS)

    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    print("Loading shared model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    attention_mask = torch.ones_like(input_ids)

    print(
        f"Input tokens: {input_ids.shape[1]}",
        flush=True,
    )

    print("Running one warm-up...", flush=True)

    with torch.inference_mode():
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    all_requests = []
    all_samples = []
    summaries = []

    for concurrency in CONCURRENCY_LEVELS:
        print(
            f"\nTesting concurrency={concurrency}",
            flush=True,
        )

        barrier = threading.Barrier(
            concurrency + 1
        )

        samples = []
        stop_event = threading.Event()

        monitor_thread = threading.Thread(
            target=monitor_resources,
            args=(
                stop_event,
                samples,
                concurrency,
            ),
            daemon=True,
        )

        monitor_thread.start()

        results = []
        errors = []

        try:
            with ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:

                futures = [
                    executor.submit(
                        run_request,
                        request_id,
                        model,
                        tokenizer,
                        input_ids,
                        attention_mask,
                        barrier,
                    )
                    for request_id in range(
                        1,
                        concurrency + 1
                    )
                ]

                barrier.wait()
                batch_start = time.perf_counter()

                for future in as_completed(futures):
                    try:
                        results.append(
                            future.result()
                        )
                    except Exception as exc:
                        errors.append(str(exc))

                batch_end = time.perf_counter()

        finally:
            stop_event.set()
            monitor_thread.join()

        if not samples:
            raise RuntimeError(
                f"No resource samples collected "
                f"for concurrency={concurrency}"
            )

        batch_seconds = (
            batch_end - batch_start
        )

        total_output_tokens = sum(
            item["output_tokens"]
            for item in results
        )

        for item in results:
            item["concurrency"] = concurrency
            all_requests.append(item)

        all_samples.extend(samples)

        latencies = [
            item["latency_seconds"]
            for item in results
        ]

        summary = {
            "concurrency": concurrency,
            "successful_requests": len(results),
            "failed_requests": len(errors),
            "batch_seconds": round(
                batch_seconds,
                4,
            ),
            "total_output_tokens":
                total_output_tokens,
            "aggregate_output_tokens_per_second":
                round(
                    total_output_tokens /
                    batch_seconds,
                    4,
                ),
            "mean_request_latency_seconds":
                round(
                    statistics.mean(latencies),
                    4,
                ) if latencies else None,
            "min_request_latency_seconds":
                round(
                    min(latencies),
                    4,
                ) if latencies else None,
            "max_request_latency_seconds":
                round(
                    max(latencies),
                    4,
                ) if latencies else None,
            "mean_process_cpu_normalized_percent":
                round(
                    statistics.mean(
                        sample[
                            "process_cpu_normalized_percent"
                        ]
                        for sample in samples
                    ),
                    2,
                ),
            "peak_process_cpu_normalized_percent":
                round(
                    max(
                        sample[
                            "process_cpu_normalized_percent"
                        ]
                        for sample in samples
                    ),
                    2,
                ),
            "mean_host_cpu_percent":
                round(
                    statistics.mean(
                        sample["host_cpu_percent"]
                        for sample in samples
                    ),
                    2,
                ),
            "peak_rss_mib":
                round(
                    max(
                        sample["process_rss_mib"]
                        for sample in samples
                    ),
                    2,
                ),
            "errors": errors,
        }

        summaries.append(summary)

        print(
            f"  Successful: "
            f"{summary['successful_requests']}",
            flush=True,
        )

        print(
            f"  Mean latency: "
            f"{summary['mean_request_latency_seconds']} s",
            flush=True,
        )

        print(
            f"  Batch time: "
            f"{summary['batch_seconds']} s",
            flush=True,
        )

        print(
            f"  Aggregate throughput: "
            f"{summary['aggregate_output_tokens_per_second']} tok/s",
            flush=True,
        )

        print(
            f"  Mean CPU: "
            f"{summary['mean_process_cpu_normalized_percent']}%",
            flush=True,
        )

        print(
            f"  Peak RSS: "
            f"{summary['peak_rss_mib']} MiB",
            flush=True,
        )

    evidence = {
        "phase": "9.6",
        "measurement":
            "concurrency_capacity_test",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "hostname": platform.node(),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "device": "cpu",
        "precision": "float32",
        "logical_cpus": logical_cpus,
        "torch_threads":
            torch.get_num_threads(),
        "input_tokens":
            int(input_ids.shape[1]),
        "max_new_tokens":
            MAX_NEW_TOKENS,
        "concurrency_levels":
            CONCURRENCY_LEVELS,
        "summaries":
            summaries,
        "requests":
            all_requests,
    }

    json_path = (
        OUTPUT_DIR /
        "concurrency-capacity.json"
    )

    request_csv = (
        OUTPUT_DIR /
        "concurrency-requests.csv"
    )

    sample_csv = (
        OUTPUT_DIR /
        "concurrency-resource-samples.csv"
    )

    json_path.write_text(
        json.dumps(
            evidence,
            indent=2,
        ) + "\n"
    )

    with request_csv.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                all_requests[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(all_requests)

    with sample_csv.open(
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
        "========== PHASE 9.6 SUMMARY =========="
    )

    print(
        json.dumps(
            summaries,
            indent=2,
        )
    )

    print()
    print("Saved:", json_path)
    print("Saved:", request_csv)
    print("Saved:", sample_csv)


if __name__ == "__main__":
    main()
