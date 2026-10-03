export OLLAMA_DUMMY_KEY=ollama
python -m baselines.harness_baseline --harness opencode --iac-type cloudformation \
  --dataset data/cfn_eval_benchmark_real_aws.csv --rows "130,131,152,154,158,159,162,166,168,170,174,175,179,200,215,229,230" --deploy-target aws --max-iterations 15 \
  --model deepseek-v4-flash:cloud --base-url http://localhost:11434 --api-key-env OLLAMA_DUMMY_KEY \
  --provider-name ollama --no-retry-proxy --scenario-timeout 14400 \
  --runs-dir runs/OpencodeDSV4F_CFNEvalRealAWS_Full_runs \
  --output-dir benchmark_runs/baseline_opencode_cloudformation_20260917_232736_DSV4F_Full/batch4 --keep-workspace
