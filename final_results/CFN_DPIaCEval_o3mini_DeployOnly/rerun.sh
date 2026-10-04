# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type cloudformation --dataset data/iac_with_difficulty_levels.csv --rows "114,147" --provider openrouter --model openai/o3-mini --deploy-target aws --max-iterations 15 --no-max-tokens \
  --output-dir benchmark_runs/cloudformation_20260922_232123_DPIaCEval_O3Mini/rerun_$(date +%Y%m%d_%H%M%S)
