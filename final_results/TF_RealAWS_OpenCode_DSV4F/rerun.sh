# harness run: fill in the provider flags you used originally (see the experiment's Notion page)
python -m baselines.harness_baseline --harness opencode --iac-type terraform --dataset data/tf_eval_benchmark_real_aws.csv --rows "249" --deploy-target aws --max-iterations 15 --model deepseek-v4-flash --no-retry-proxy --scenario-timeout 14400 --keep-workspace \
  --output-dir benchmark_runs/baseline_opencode_terraform_20260925_104635/rerun_$(date +%Y%m%d_%H%M%S)
