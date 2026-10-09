# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type cloudformation --dataset data/cfn_eval_benchmark_real_aws.csv --rows "6,198,257" --provider openai --model claude-opus-5.5 --deploy-target aws --max-iterations 1 --no-max-tokens \
  --output-dir benchmark_runs/opus_mas1_cloudformation_20261008_182531/rerun_$(date +%Y%m%d_%H%M%S)
