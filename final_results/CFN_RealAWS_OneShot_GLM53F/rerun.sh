# reconstructed from summary.json of the first run folder; check against how you launched it
python -m baselines.oneshot_baseline --iac-type cloudformation --dataset data/cfn_eval_benchmark_real_aws.csv --rows "40,45,130,250" --provider openrouter --model z-ai/glm-5.3-flash --deploy-target aws --openrouter-reasoning-max-tokens "3000" --no-max-tokens \
  --output-dir benchmark_runs/oneshot_cloudformation_20261007_203259/rerun_$(date +%Y%m%d_%H%M%S)
