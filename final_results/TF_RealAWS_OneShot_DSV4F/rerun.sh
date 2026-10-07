# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type terraform --dataset data/tf_eval_benchmark_real_aws.csv --rows "54,84,332,344" --provider deepseek --model deepseek-v4-flash --deploy-target aws --max-iterations 1 --no-max-tokens \
  --output-dir benchmark_runs/oneshot_terraform_20261005_112650/rerun_$(date +%Y%m%d_%H%M%S)
