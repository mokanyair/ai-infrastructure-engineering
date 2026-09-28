#!/usr/bin/env python3

import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"

PROMPT = """<|system|>
You are a concise AI infrastructure assistant.</s>
<|user|>
Explain in one paragraph why GPU utilization matters for AI infrastructure capacity planning.</s>
<|assistant|>
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Week 2 TinyLlama GPU workload-shape benchmark"
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        required=True
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8]
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--measured-runs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--output-dir",
        default="results"
    )

    return parser.parse_args()


def dtype_for_precision(precision):
    if precision == "fp32":
        return torch.float32

    if precision == "fp16":
        return torch.float16

    raise ValueError(f"Unsupported precision: {precision}")


def mib(value):
    return round(value / (1024 ** 2), 2)


def gpu_memory():
    return {
        "allocated_mib": mib(torch.cuda.memory_allocated()),
        "reserved_mib": mib(torch.cuda.memory_reserved()),
        "max_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "max_reserved_mib": mib(torch.cuda.max_memory_reserved()),
    }


def system_info():
    props = torch.cuda.get_device_properties(0)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "gpu_vram_gib": round(props.total_memory / 1024**3, 2),
        "gpu_sm_count": props.multi_processor_count,
        "cpu_logical": psutil.cpu_count(logical=True),
        "system_ram_gib": round(psutil.virtual_memory().total / 1024**3, 2),
    }


def synchronize():
    torch.cuda.synchronize()


def run_generation(model, tokenizer, batch_size, max_new_tokens):
    prompts = [PROMPT] * batch_size

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True
    )

    encoded = {
        key: value.to("cuda")
        for key, value in encoded.items()
    }

    input_tokens_per_request = encoded["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()

    synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )

    synchronize()
    elapsed = time.perf_counter() - start

    total_input_tokens = input_tokens_per_request * batch_size
    total_output_tokens = 0

    for i in range(batch_size):
        total_output_tokens += (
            outputs[i].shape[0] - input_tokens_per_request
        )

    output_tokens_per_second = (
        total_output_tokens / elapsed
        if elapsed > 0 else 0
    )

    requests_per_second = (
        batch_size / elapsed
        if elapsed > 0 else 0
    )

    return {
        "batch_size": batch_size,
        "latency_seconds": round(elapsed, 4),
        "input_tokens_total": total_input_tokens,
        "output_tokens_total": total_output_tokens,
        "output_tokens_per_second": round(output_tokens_per_second, 4),
        "requests_per_second": round(requests_per_second, 4),
        **gpu_memory()
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: CUDA is not available. Run runtime validation first."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = dtype_for_precision(args.precision)

    print("=" * 70)
    print("WEEK 2 GPU WORKLOAD-SHAPE BENCHMARK")
    print("=" * 70)
    print("Model:", MODEL_ID)
    print("Revision:", MODEL_REVISION)
    print("Precision:", args.precision)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Batch sizes:", args.batch_sizes)
    print()

    process = psutil.Process(os.getpid())

    rss_before = process.memory_info().rss

    tokenizer_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION
    )

    tokenizer_load_seconds = time.perf_counter() - tokenizer_start

    # Decoder-only batched generation requires padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    model.eval()
    model.to("cuda")

    synchronize()

    model_load_seconds = time.perf_counter() - model_start

    rss_after = process.memory_info().rss

    model_memory = gpu_memory()

    print(
        f"Tokenizer load: {tokenizer_load_seconds:.3f}s"
    )
    print(
        f"Model load: {model_load_seconds:.3f}s"
    )
    print(
        f"GPU allocated after model load: "
        f"{model_memory['allocated_mib']} MiB"
    )
    print()

    results = []

    for batch_size in args.batch_sizes:
        print(f"--- Batch size {batch_size} ---")

        # Warmup removes first-execution effects from measured runs.
        for _ in range(args.warmup_runs):
            run_generation(
                model,
                tokenizer,
                batch_size,
                args.max_new_tokens
            )

        batch_results = []

        for run_number in range(1, args.measured_runs + 1):
            result = run_generation(
                model,
                tokenizer,
                batch_size,
                args.max_new_tokens
            )

            result["run"] = run_number
            result["precision"] = args.precision

            batch_results.append(result)
            results.append(result)

            print(
                f"run={run_number} "
                f"latency={result['latency_seconds']}s "
                f"tokens/s={result['output_tokens_per_second']} "
                f"peak_vram={result['max_allocated_mib']} MiB"
            )

        print(
            "mean latency:",
            round(
                statistics.mean(
                    x["latency_seconds"]
                    for x in batch_results
                ),
                4
            ),
            "seconds"
        )

        print(
            "mean tokens/sec:",
            round(
                statistics.mean(
                    x["output_tokens_per_second"]
                    for x in batch_results
                ),
                4
            )
        )

        print()

    metadata = {
        "experiment": "week02-gpu-workload-shape",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "precision": args.precision,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "tokenizer_load_seconds": round(
            tokenizer_load_seconds, 4
        ),
        "model_load_seconds": round(
            model_load_seconds, 4
        ),
        "rss_before_model_mib": mib(rss_before),
        "rss_after_model_mib": mib(rss_after),
        "gpu_memory_after_model_load": model_memory,
        "system": system_info(),
        "results": results
    }

    json_path = (
        output_dir /
        f"benchmark_{args.precision}.json"
    )

    csv_path = (
        output_dir /
        f"benchmark_{args.precision}.csv"
    )

    with json_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    fieldnames = [
        "precision",
        "batch_size",
        "run",
        "latency_seconds",
        "input_tokens_total",
        "output_tokens_total",
        "output_tokens_per_second",
        "requests_per_second",
        "allocated_mib",
        "reserved_mib",
        "max_allocated_mib",
        "max_reserved_mib"
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()

        for row in results:
            writer.writerow({
                key: row[key]
                for key in fieldnames
            })

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("JSON:", json_path)
    print("CSV :", csv_path)


if __name__ == "__main__":
    main()
