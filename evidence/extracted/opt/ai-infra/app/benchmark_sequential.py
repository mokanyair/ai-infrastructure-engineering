import csv
import json
import os
import platform
import statistics
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

WARMUP_RUNS = 2
MEASURED_RUNS = 5
MAX_NEW_TOKENS = 64

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


def rss_mib():
    return round(
        process.memory_info().rss / (1024 ** 2),
        2,
    )


def run_inference(model, tokenizer, inputs, attention_mask):
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

    generated_ids = outputs[
        0,
        inputs.shape[1]:,
    ]

    output_tokens = generated_ids.numel()

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    if not response or output_tokens <= 1:
        raise RuntimeError(
            "Invalid inference output."
        )

    return {
        "latency_seconds": round(elapsed, 4),
        "output_tokens": output_tokens,
        "tokens_per_second": round(
            output_tokens / elapsed,
            4,
        ),
        "process_rss_mib": rss_mib(),
        "response_nonempty": bool(response),
    }


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

    print("Running warm-up requests...", flush=True)

    for i in range(WARMUP_RUNS):
        result = run_inference(
            model,
            tokenizer,
            inputs,
            attention_mask,
        )

        print(
            f"Warm-up {i + 1}: "
            f"{result['latency_seconds']} s",
            flush=True,
        )

    print("Running measured requests...", flush=True)

    measurements = []

    for i in range(MEASURED_RUNS):
        result = run_inference(
            model,
            tokenizer,
            inputs,
            attention_mask,
        )

        result["run"] = i + 1

        measurements.append(result)

        print(
            f"Run {i + 1}: "
            f"{result['latency_seconds']} s | "
            f"{result['tokens_per_second']} tokens/s",
            flush=True,
        )

    latencies = [
        item["latency_seconds"]
        for item in measurements
    ]

    throughputs = [
        item["tokens_per_second"]
        for item in measurements
    ]

    summary = {
        "phase": "9.3",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "hostname": platform.node(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": "cpu",
        "dtype": "torch.float32",
        "cpu_threads": torch.get_num_threads(),
        "input_tokens": inputs.shape[1],
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "mean_latency_seconds": round(
            statistics.mean(latencies), 4
        ),
        "median_latency_seconds": round(
            statistics.median(latencies), 4
        ),
        "min_latency_seconds": min(latencies),
        "max_latency_seconds": max(latencies),
        "mean_tokens_per_second": round(
            statistics.mean(throughputs), 4
        ),
        "measurements": measurements,
    }

    json_path = OUTPUT_DIR / (
        "sequential-benchmark.json"
    )

    csv_path = OUTPUT_DIR / (
        "sequential-benchmark.csv"
    )

    json_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                measurements[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(measurements)

    print()
    print("========== BENCHMARK SUMMARY ==========")
    print(json.dumps(summary, indent=2))

    print()
    print("Saved JSON:", json_path)
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
