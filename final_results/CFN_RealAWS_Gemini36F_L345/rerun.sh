# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type cloudformation --dataset data/cfn_eval_benchmark_real_aws_diff345.csv --rows "250,251,252,253,254,256,257,258,259,262,265,266,268" --provider openrouter --model google/gemini-3.6-flash --deploy-target aws --max-iterations 15 --openrouter-provider-only "google-vertex/global/flex,google-ai-studio/flex" --openrouter-reasoning-max-tokens "3000" \
  --output-dir benchmark_runs/cloudformation_20260915_154934_Gemini36Flash/rerun_$(date +%Y%m%d_%H%M%S)
