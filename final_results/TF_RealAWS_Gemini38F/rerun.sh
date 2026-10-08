# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type terraform --dataset data/tf_eval_benchmark_real_aws.csv --rows "249" --provider openrouter --model google/gemini-3.8-flash --deploy-target aws --max-iterations 15 --openrouter-provider-only "google-ai-studio/flex,google-vertex/global/flex" --openrouter-reasoning-max-tokens "3000" --no-max-tokens \
  --output-dir benchmark_runs/terraform_20260929_151853_Gemini38F/rerun_$(date +%Y%m%d_%H%M%S)
