# reconstructed from summary.json of the first run folder; check against how you launched it
python benchmark.py --iac-type cloudformation --dataset data/cfn_eval_benchmark_real_aws_diff345.csv --rows "154,175,228,230,232,250,251,252,253,254,256,257,258,259,262,265,266,268" --provider openrouter --model deepseek/deepseek-v4-flash --deploy-target aws --max-iterations 15 --openrouter-reasoning-max-tokens "3000" \
  --output-dir benchmark_runs/cloudformation_20260908_115729_DSV4F_Dep/rerun_$(date +%Y%m%d_%H%M%S)
