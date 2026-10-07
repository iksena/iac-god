# harness run: fill in the provider flags you used originally (see the experiment's Notion page)
python -m baselines.harness_baseline --harness opencode --iac-type cloudformation --dataset data/cfn_eval_benchmark_ablation.csv --rows "47,93,240" --deploy-target localstack --max-iterations 15 --model deepseek/deepseek-v4-flash --no-retry-proxy --scenario-timeout 14400 --keep-workspace \
  --output-dir benchmark_runs/baseline_opencode_retriever_cloudformation_20260919_173443_Ablation_HarnessAndRetriever/rerun_$(date +%Y%m%d_%H%M%S)
