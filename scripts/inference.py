import json
import os
import platform
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = os.environ["MODEL_REVISION"]
CACHE_DIR = os.environ["HF_HOME"]

MESSAGES = [
    {
        "role": "user",
        "content": (
            "Explain in two short sentences why infrastructure engineers "
            "measure CPU utilization and memory usage during LLM inference."
        ),
    }
]

MAX_NEW_TOKENS = 64


def memory_mb():
    """Return the current process resident memory in MiB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    # Keep the initial CPU configuration consistent.
    torch.set_num_threads(4)

    print("Loading tokenizer...", flush=True)

    tokenizer_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    tokenizer_load_seconds = (
        time.perf_counter() - tokenizer_start
    )

    # Prepare the prompt using TinyLlama's chat template.
    inputs = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    input_tokens = inputs.shape[1]

    print("Loading model...", flush=True)

    rss_before = memory_mb()
    model_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    model.to("cpu")
    model.eval()

    model_load_seconds = (
        time.perf_counter() - model_start
    )

    rss_after = memory_mb()

    parameter = next(model.parameters())

    # Fail immediately if the experiment configuration is wrong.
    if parameter.device.type != "cpu":
        raise RuntimeError(
            f"Expected CPU execution, got {parameter.device}"
        )

    if parameter.dtype != torch.float32:
        raise RuntimeError(
            f"Expected FP32, got {parameter.dtype}"
        )

    print("Running inference...", flush=True)

    inference_start = time.perf_counter()

    attention_mask = torch.ones_like(inputs)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    # Remove the original input tokens from the generated sequence.
    output_ids = outputs[0, input_tokens:]

    generated_tokens = output_ids.numel()

    special_token_ids = set(tokenizer.all_special_ids)

    content_tokens = sum(
        token_id not in special_token_ids
        for token_id in output_ids.tolist()
    )

    response = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    ).strip()

    tokens_per_second = (
        generated_tokens / inference_seconds
        if inference_seconds > 0
        else 0.0
    )

    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "tokenizer_load_seconds": round(
            tokenizer_load_seconds, 3
        ),
        "model_load_seconds": round(
            model_load_seconds, 3
        ),
        "rss_before_model_mb": round(
            rss_before, 2
        ),
        "rss_after_model_mb": round(
            rss_after, 2
        ),
        "input_tokens": int(input_tokens),
        "output_tokens": int(generated_tokens),
        "content_tokens": int(content_tokens),
        "inference_seconds": round(
            inference_seconds, 3
        ),
        "output_tokens_per_second": round(
            tokens_per_second, 3
        ),
        "response": response,
    }

    print(json.dumps(result, indent=2), flush=True)

    # A nonempty response is required to pass the functional test.
    if not response:
        raise RuntimeError(
            "Functional inference failed: generated response is empty."
        )


if __name__ == "__main__":
    main()
