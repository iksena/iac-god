# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type cloudformation --dataset data/cfn_eval_benchmark_real_aws.csv --rows "1,6,13,56,74,213" --provider deepseek --model deepseek-v4-flash --deploy-target aws --max-iterations 1 \
  --output-dir benchmark_runs/oneshot_cloudformation_20261003_220413_DeepseekV4Flash/rerun_$(date +%Y%m%d_%H%M%S)
