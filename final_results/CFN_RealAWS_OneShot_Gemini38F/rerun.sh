# reconstructed from summary.json of the first run folder; check against how you launched it
python -m baselines.oneshot_baseline --iac-type cloudformation --dataset data/cfn_eval_benchmark_real_aws.csv --rows "1,6,13,56,74,103" --provider openrouter --model google/gemini-3.8-flash --deploy-target aws --openrouter-provider-only "google-ai-studio/flex,google-vertex/global/flex" --openrouter-reasoning-max-tokens "3000" --no-max-tokens \
  --output-dir benchmark_runs/oneshot_cloudformation_20261006_172747/rerun_$(date +%Y%m%d_%H%M%S)
