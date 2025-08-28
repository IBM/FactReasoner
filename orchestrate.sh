# !/bin/bash
# This script illustrates the orchestrate commands for running all the experiments
# performed so far. Note that in real use, it would be advisable to split the
# experiments into smaller batches to avoid overloading RITS/CCC.
python3 orchestrate.py \
    --datasets wiki_contradict_humaneval conflict_bank \
    --models llama-3.3-70b-instruct Qwen2.5-72B-Instruct llama-4-maverick-17b-128e-instruct-fp8 \
    --variants qa nli e2e \
    --relevance_thresholds 3.5 \
    --confidence_thresholds 2.0 \
    --experiment_version v12 \
    --enable_tools True False \
    --use_default_prompt_template False \
    --env_name experiments \
    --repo_path /u/adamdejl/fm-factual
# gpt-oss models don't yet support tool calls on RITS
python3 orchestrate.py \
    --datasets wiki_contradict_humaneval conflict_bank \
    --models gpt-oss-20b gpt-oss-120b \
    --variants qa nli e2e \
    --relevance_thresholds 3.5 \
    --confidence_thresholds 2.0 \
    --experiment_version v12 \
    --enable_tools False \
    --use_default_prompt_template False \
    --env_name experiments \
    --repo_path /u/adamdejl/fm-factual
python3 orchestrate.py \
    --datasets eli5_base eli5_v2 \
    --models gpt-oss-20b \
    --evaluated_models llama-3.3-70b-instruct Qwen2.5-72B-Instruct llama-4-maverick-17b-128e-instruct-fp8 gpt-oss-20b gpt-oss-120b \
    --variants qa \
    --relevance_thresholds 3.5 \
    --confidence_thresholds 2.0 \
    --experiment_version eval_ELI5 \
    --enable_tools False \
    --use_default_prompt_template False \
    --env_name experiments \
    --repo_path /u/adamdejl/fm-factual
python3 orchestrate.py \
    --datasets eli5_base eli5_v2 \
    --models llama-4-maverick-17b-128e-instruct-fp8 \
    --evaluated_models llama-3.3-70b-instruct Qwen2.5-72B-Instruct llama-4-maverick-17b-128e-instruct-fp8 gpt-oss-20b gpt-oss-120b \
    --variants e2e \
    --relevance_thresholds 3.5 \
    --confidence_thresholds 2.0 \
    --experiment_version eval_ELI5 \
    --enable_tools False \
    --use_default_prompt_template False \
    --env_name experiments \
    --repo_path /u/adamdejl/fm-factual
python3 orchestrate.py \
    --datasets eli5_base eli5_v2 \
    --models llama-3.3-70b-instruct \
    --evaluated_models llama-3.3-70b-instruct Qwen2.5-72B-Instruct llama-4-maverick-17b-128e-instruct-fp8 gpt-oss-20b gpt-oss-120b \
    --variants e2e-base \
    --relevance_thresholds 3.5 \
    --confidence_thresholds 2.0 \
    --experiment_version eval_ELI5 \
    --enable_tools False \
    --use_default_prompt_template False \
    --env_name experiments \
    --repo_path /u/adamdejl/fm-factual