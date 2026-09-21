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
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ["MODEL_ID"]
REVISION = os.environ["MODEL_REVISION"]
CACHE_DIR = os.environ["HF_HOME"]

OUTPUT = Path("/var/lib/ai-infra/benchmarks/phase9")
OUTPUT.mkdir(parents=True, exist_ok=True)

THREADS = 4
WARMUPS = 1
REPEATS = 3
SAMPLE_INTERVAL = 0.2

BASE_PROMPT = (
    "Explain in two short sentences why infrastructure "
    "engineers measure CPU utilization and memory usage "
    "during LLM inference."
)

EXTRA_CONTEXT = (
    "A production inference platform requires predictable "
    "latency, throughput, CPU capacity, memory headroom, "
    "security, observability and cost management. "
)

process = psutil.Process()
logical_cpus = psutil.cpu_count(logical=True)


def mib(value):
    return round(value / 1048576, 2)


def prepare_prompt(tokenizer, target_tokens):
    if target_tokens is None:
        prompt = BASE_PROMPT
    else:
        prompt = BASE_PROMPT

        # Grow the synthetic prompt until the chat-formatted
        # input reaches the requested approximate token length.
        for _ in range(100):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )

            if len(ids) >= target_tokens:
                break

            prompt += " " + EXTRA_CONTEXT
        else:
            raise RuntimeError(
                "Unable to construct long synthetic prompt."
            )

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    return inputs, torch.ones_like(inputs)


def sample_resources(stop, samples):
    process.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)

    start = time.perf_counter()

    while not stop.is_set():
        cpu = process.cpu_percent(
            interval=SAMPLE_INTERVAL
        )
        host_cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        samples.append({
            "elapsed_seconds": round(
                time.perf_counter() - start, 4
            ),
            "process_cpu_raw_percent": round(cpu, 2),
            "process_cpu_normalized_percent": round(
                cpu / logical_cpus, 2
            ),
            "host_cpu_percent": round(host_cpu, 2),
            "process_rss_mib": mib(
                process.memory_info().rss
            ),
            "host_available_mib": mib(
                memory.available
            ),
        })


def generate(model, tokenizer, inputs, mask, max_tokens):
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency = time.perf_counter() - start
    generated = output[0, inputs.shape[1]:]
    tokens = generated.numel()

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    if tokens <= 1 or not response:
        raise RuntimeError(
            "Inference produced invalid or empty output."
        )

    return latency, tokens


def measured_run(
    model, tokenizer, inputs, mask,
    max_tokens, experiment, run_number
):
    samples = []
    stop = threading.Event()

    monitor = threading.Thread(
        target=sample_resources,
        args=(stop, samples),
        daemon=True,
    )

    monitor.start()

    try:
        latency, tokens = generate(
            model, tokenizer, inputs,
            mask, max_tokens
        )
    finally:
        stop.set()
        monitor.join()

    if not samples:
        raise RuntimeError(
            "Resource monitoring collected no samples."
        )

    result = {
        "experiment": experiment,
        "run": run_number,
        "input_tokens": int(inputs.shape[1]),
        "output_ceiling": max_tokens,
        "actual_output_tokens": int(tokens),
        "latency_seconds": round(latency, 4),
        "generation_tokens_per_second": round(
            tokens / latency, 4
        ),
        "mean_cpu_normalized_percent": round(
            statistics.mean(
                x["process_cpu_normalized_percent"]
                for x in samples
            ), 2
        ),
        "peak_cpu_normalized_percent": round(
            max(
                x["process_cpu_normalized_percent"]
                for x in samples
            ), 2
        ),
        "peak_rss_mib": max(
            x["process_rss_mib"] for x in samples
        ),
        "min_host_available_mib": min(
            x["host_available_mib"] for x in samples
        ),
        "resource_samples": len(samples),
    }

    return result, samples


def main():
    torch.set_num_threads(THREADS)

    print("Loading tokenizer and model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()

    experiments = [
        ("A_baseline", None, 64),
        ("B_long_input", 256, 64),
        ("C_long_output", None, 128),
        ("D_combined", 256, 128),
    ]

    all_results = []
    all_samples = []
    summaries = []

    for name, target_input, output_limit in experiments:
        inputs, mask = prepare_prompt(
            tokenizer, target_input
        )

        print(
            f"\n{name}: input={inputs.shape[1]}, "
            f"output ceiling={output_limit}",
            flush=True,
        )

        for warmup in range(WARMUPS):
            generate(
                model, tokenizer, inputs,
                mask, output_limit
            )

        experiment_results = []

        for run in range(1, REPEATS + 1):
            result, samples = measured_run(
                model, tokenizer, inputs,
                mask, output_limit, name, run
            )

            experiment_results.append(result)
            all_results.append(result)

            for sample in samples:
                sample["experiment"] = name
                sample["run"] = run
                all_samples.append(sample)

            print(
                f"  Run {run}: "
                f"{result['latency_seconds']} s | "
                f"{result['actual_output_tokens']} tokens | "
                f"{result['generation_tokens_per_second']} tok/s",
                flush=True,
            )

        summary = {
            "experiment": name,
            "input_tokens": int(inputs.shape[1]),
            "output_ceiling": output_limit,
            "mean_actual_output_tokens": round(
                statistics.mean(
                    r["actual_output_tokens"]
                    for r in experiment_results
                ), 2
            ),
            "mean_latency_seconds": round(
                statistics.mean(
                    r["latency_seconds"]
                    for r in experiment_results
                ), 4
            ),
            "mean_generation_tokens_per_second": round(
                statistics.mean(
                    r["generation_tokens_per_second"]
                    for r in experiment_results
                ), 4
            ),
            "mean_cpu_normalized_percent": round(
                statistics.mean(
                    r["mean_cpu_normalized_percent"]
                    for r in experiment_results
                ), 2
            ),
            "peak_rss_mib": max(
                r["peak_rss_mib"]
                for r in experiment_results
            ),
        }

        summaries.append(summary)

    evidence = {
        "phase": "9.5",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "hostname": platform.node(),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "device": "cpu",
        "precision": "float32",
        "logical_cpus": logical_cpus,
        "torch_threads": torch.get_num_threads(),
        "warmups_per_experiment": WARMUPS,
        "measured_runs_per_experiment": REPEATS,
        "sampling_interval_target_seconds":
            SAMPLE_INTERVAL,
        "experiment_summaries": summaries,
        "runs": all_results,
    }

    json_path = OUTPUT / "workload-scaling.json"
    csv_path = OUTPUT / "workload-scaling.csv"
    samples_path = OUTPUT / "workload-scaling-samples.csv"

    json_path.write_text(
        json.dumps(evidence, indent=2) + "\n"
    )

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_results)

    with samples_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_samples[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_samples)

    print("\n========== PHASE 9.5 SUMMARY ==========")
    print(json.dumps(summaries, indent=2))
    print("\nSaved:", json_path)
    print("Saved:", csv_path)
    print("Saved:", samples_path)


if __name__ == "__main__":
    main()
