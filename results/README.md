# Benchmark Results

Raw benchmark files are generated during experiments and are not committed
by default.

Each benchmark record should contain:

- run_id
- timestamp
- instance_type
- model_id
- model_revision
- input_tokens
- output_tokens
- concurrency
- request_latency_ms
- queue_wait_ms
- execution_time_ms
- tokens_per_second
- success
- error
- cpu_percent
- memory_percent
- process_rss_mb
