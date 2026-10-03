# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type terraform --dataset data/iac_eval_benchmark.csv --rows "60,71,87,133,139" --provider openrouter --model deepseek/deepseek-v4-flash --deploy-target none --max-iterations 15 --skip-security \
  --output-dir benchmark_runs/terraform_20260908_171434_IaCEval/rerun_$(date +%Y%m%d_%H%M%S)
