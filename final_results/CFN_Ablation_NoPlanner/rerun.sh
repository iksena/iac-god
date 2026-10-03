# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type cloudformation --dataset data/cfn_eval_benchmark_ablation.csv --rows "85" --provider deepseek --model deepseek-v4-flash --deploy-target localstack --max-iterations 15 --no-max-tokens \
  --output-dir benchmark_runs/cloudformation_20260930_130454_Ablation_NoPlanner/rerun_$(date +%Y%m%d_%H%M%S)
