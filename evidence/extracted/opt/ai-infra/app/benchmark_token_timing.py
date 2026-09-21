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

MODEL_ID = os.environ["MODEL_ID"]
REVISION = os.environ["MODEL_REVISION"]
CACHE_DIR = os.environ["HF_HOME"]

OUTPUT_DIR = Path("/var/lib/ai-infra/benchmarks/phase9")

THREADS = 4
WARMUPS = 2
MEASURED_RUNS = 5
MAX_NEW_TOKENS = 64

PROMPT = (
    "Explain in two short sentences why infrastructure "
    "engineers measure CPU utilization and memory usage "
    "during LLM inference."
)


def run_generation(model, tokenizer, input_ids, attention_mask):
    """
    Greedy decoding with KV-cache reuse.

    First forward pass: full prompt.
    Later passes: one new token plus past_key_values.
    """

    eos_id = tokenizer.eos_token_id
    generated = []
    token_timestamps = []

    start = time.perf_counter()

    with torch.inference_mode():

        # PREFILL: process the entire input prompt.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        token_timestamps.append(time.perf_counter())
        generated.append(int(next_token.item()))

        # DECODE: feed only the most recent token.
        for _ in range(MAX_NEW_TOKENS - 1):

            if generated[-1] == eos_id:
                break

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )

            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            token_timestamps.append(time.perf_counter())
            generated.append(int(next_token.item()))

    if len(generated) < 2:
        raise RuntimeError(
            "Insufficient tokens for decode throughput measurement."
        )

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    if not response:
        raise RuntimeError("Generated response is empty.")

    first_token_time = token_timestamps[0]
    last_token_time = token_timestamps[-1]

    ttft = first_token_time - start
    decode_duration = last_token_time - first_token_time
    total_latency = last_token_time - start

    if decode_duration <= 0:
        raise RuntimeError("Invalid decode duration.")

    output_tokens = len(generated)
    decode_tokens = output_tokens - 1

    return {
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": output_tokens,
        "ttft_seconds": round(ttft, 4),
        "decode_tokens": decode_tokens,
        "decode_seconds": round(decode_duration, 4),
        "pure_decode_tokens_per_second": round(
            decode_tokens / decode_duration, 4
        ),
        "end_to_end_latency_seconds": round(
            total_latency, 4
        ),
        "effective_output_tokens_per_second": round(
            output_tokens / total_latency, 4
        ),
        "stopped_at_eos": generated[-1] == eos_id,
        "peak_process_rss_mib": round(
            psutil.Process().memory_info().rss / (1024 ** 2),
            2,
        ),
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

    print("Loading model...", flush=True)

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

    print(
        f"Warm-ups: {WARMUPS}; measured runs: {MEASURED_RUNS}",
        flush=True,
    )

    for warmup in range(1, WARMUPS + 1):
        run_generation(
            model,
            tokenizer,
            input_ids,
            attention_mask,
        )
        print(f"Warm-up {warmup}: complete", flush=True)

    results = []

    for run in range(1, MEASURED_RUNS + 1):
        result = run_generation(
            model,
            tokenizer,
            input_ids,
            attention_mask,
        )

        result["run"] = run
        results.append(result)

        print(
            f"Run {run}: "
            f"TTFT={result['ttft_seconds']} s | "
            f"decode={result['pure_decode_tokens_per_second']} tok/s | "
            f"latency={result['end_to_end_latency_seconds']} s",
            flush=True,
        )

    summary = {
        "mean_ttft_seconds": round(
            statistics.mean(
                r["ttft_seconds"] for r in results
            ),
            4,
        ),
        "median_ttft_seconds": round(
            statistics.median(
                r["ttft_seconds"] for r in results
            ),
            4,
        ),
        "mean_pure_decode_tokens_per_second": round(
            statistics.mean(
                r["pure_decode_tokens_per_second"]
                for r in results
            ),
            4,
        ),
        "mean_decode_seconds": round(
            statistics.mean(
                r["decode_seconds"] for r in results
            ),
            4,
        ),
        "mean_end_to_end_latency_seconds": round(
            statistics.mean(
                r["end_to_end_latency_seconds"]
                for r in results
            ),
            4,
        ),
        "mean_effective_output_tokens_per_second": round(
            statistics.mean(
                r["effective_output_tokens_per_second"]
                for r in results
            ),
            4,
        ),
        "output_tokens_per_run": [
            r["output_tokens"] for r in results
        ],
    }

    evidence = {
        "phase": "9.7",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "hostname": platform.node(),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "device": "cpu",
        "precision": "float32",
        "torch_threads": torch.get_num_threads(),
        "warmups": WARMUPS,
        "measured_runs": MEASURED_RUNS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "method": (
            "Manual greedy decoding with KV-cache reuse. "
            "First forward pass processes full prompt; "
            "subsequent passes process one token."
        ),
        "summary": summary,
        "runs": results,
    }

    json_path = OUTPUT_DIR / "token-timing.json"
    csv_path = OUTPUT_DIR / "token-timing.csv"

    json_path.write_text(
        json.dumps(evidence, indent=2) + "\n"
    )

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n========== PHASE 9.7 SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
