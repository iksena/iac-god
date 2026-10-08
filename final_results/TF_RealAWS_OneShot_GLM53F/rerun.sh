# reconstructed from summary.json of the first run folder; check against how you launched it
python -m baselines.oneshot_baseline --iac-type terraform --dataset data/tf_eval_benchmark_real_aws.csv --rows "13" --provider openrouter --model z-ai/glm-5.3-flash --deploy-target aws --openrouter-reasoning-max-tokens "3000" --no-max-tokens \
  --output-dir benchmark_runs/oneshot_terraform_20261008_121356/rerun_$(date +%Y%m%d_%H%M%S)
