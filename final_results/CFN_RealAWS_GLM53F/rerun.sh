# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type cloudformation --dataset data/cfn_eval_benchmark_real_aws.csv --rows "1,103,263" --provider openrouter --model z-ai/glm-5.3-flash --deploy-target aws --max-iterations 15 --openrouter-reasoning-max-tokens "3000" --no-max-tokens \
  --output-dir benchmark_runs/cloudformation_20260922_235243_GLM53Flash/rerun_$(date +%Y%m%d_%H%M%S)
