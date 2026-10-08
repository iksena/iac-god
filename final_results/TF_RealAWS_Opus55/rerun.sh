# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type terraform --dataset data/tf_eval_benchmark_real_aws.csv --rows "269,334" --provider openai --model claude-opus-5.5 --deploy-target aws --max-iterations 15 --openrouter-reasoning-max-tokens "3000" --no-max-tokens \
  --output-dir benchmark_runs/terraform_20261002_115801/rerun_$(date +%Y%m%d_%H%M%S)
