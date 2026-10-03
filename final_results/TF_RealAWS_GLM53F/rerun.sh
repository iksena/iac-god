# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type terraform --dataset data/tf_eval_benchmark_real_aws.csv --rows "161,186,187,188,189,190,192,193,194,195,197,198,199,200,209,212,223,225,249,269,286,289" --provider openrouter --model z-ai/glm-5.3-flash --deploy-target aws --max-iterations 15 --openrouter-reasoning-max-tokens "3000" --no-max-tokens \
  --output-dir benchmark_runs/terraform_20260925_102452_GLM53F/rerun_$(date +%Y%m%d_%H%M%S)
