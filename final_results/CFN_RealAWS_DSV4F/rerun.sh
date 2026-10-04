# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type cloudformation --dataset data/cfn_eval_benchmark_real_aws.csv --rows "6,20,106" --provider openrouter --model deepseek/deepseek-v4-flash --deploy-target aws --max-iterations 15 \
  --output-dir benchmark_runs/cloudformation_20260917_170146_DSV4F_Full/rerun_$(date +%Y%m%d_%H%M%S)
