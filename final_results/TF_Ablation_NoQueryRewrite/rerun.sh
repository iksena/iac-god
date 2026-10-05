# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type terraform --dataset data/tf_eval_benchmark_ablation.csv --rows "249" --provider deepseek --model deepseek-v4-flash --deploy-target localstack --max-iterations 15 --no-max-tokens \
  --output-dir benchmark_runs/terraform_20261004_192046_Ablation_NoQueryRewrite/rerun_$(date +%Y%m%d_%H%M%S)
