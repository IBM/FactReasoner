import argparse
import getpass
import itertools
import os
import regex
import stat
import subprocess
import sys
import time
import uuid

from src.fact_reasoner.comprehensiveness import DATA_LOADERS
from src.fact_reasoner.utils import RITS_MODELS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    help="The IDs of datasets for which to run experiments.",
    required=True,
    choices=DATA_LOADERS.keys(),
)
parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    help="Models to use for the experiments.",
    required=True,
    choices=RITS_MODELS.keys(),
)
parser.add_argument(
    "--evaluated_models",
    type=str,
    nargs="+",
    help="The models to be evaluated for comprehensiveness.",
    choices=RITS_MODELS.keys(),
)
parser.add_argument(
    "--variants",
    type=str,
    nargs="+",
    help="The versions of the pipeline to use.",
    required=True,
    choices=["qa", "nli", "e2e", "e2e-base"],
)
parser.add_argument(
    "--relevance_thresholds",
    type=float,
    nargs="+",
    help="The relevance thresholds to use for the experiments.",
    required=True,
)
parser.add_argument(
    "--confidence_thresholds",
    type=float,
    nargs="+",
    help="The answer confidence thresholds to use for the experiments.",
    default=[2.0],
)
parser.add_argument(
    "--experiment_version",
    type=str,
    help="The experiment version used to differentiate between different experiment runs (e.g., v1).",
    required=True,
)
parser.add_argument(
    "--enable_tools",
    type=str,
    nargs="+",
    help="Configures whether tools should be used for comparing the answers.",
    default=["True"],
)
parser.add_argument(
    "--use_default_prompt_template",
    type=str,
    nargs="+",
    help="Configures usage of the default prompt template for the chat completion API instead of injecting it manually.",
    default=["False"],
)
parser.add_argument(
    "--env_name",
    type=str,
    default="experiments",
    help="The name of the mamba environment with the dependencies needed for running the experiments.",
)
parser.add_argument(
    "--repo_path",
    type=str,
    help="The path to the main comprehensiveness repo directory.",
    required=True,
)
parser.add_argument(
    "--dry_run",
    default=False,
    action="store_true",
    help="Prints out the configuration of the experiments without submitting them on the cluster.",
)
args = parser.parse_args()

experiment_config = {
    "dataset": args.datasets,
    "model": args.models,
    "eval_model": args.evaluated_models or [None],
    "variant": args.variants,
    "relevance_threshold": args.relevance_thresholds,
    "confidence_threshold": args.confidence_thresholds,
    "use_default_prompt_template": args.use_default_prompt_template,
    "enable_tools": args.enable_tools,
}
experiment_configs = [
    dict(zip(experiment_config.keys(), choices))
    for choices in itertools.product(*experiment_config.values())
]
print(f"Planned experiment configurations ({len(experiment_configs)} in total):")
for i, config in enumerate(experiment_configs):
    print(f"{i + 1}) {config}")
print()

if args.dry_run:
    print("Dry run completed. Exiting.")
    sys.exit(0)

seconds = 30
while seconds > 0:
    print(f"Starting experiment submission in {seconds} second(s).", end="\r")
    time.sleep(1)
    seconds -= 1
print("Starting experiment submission...                ")
print()

ARG_MAPPING = {
    "eval_model": lambda em: f"--evaluated_model_name {em}" if em is not None else "",
    "use_default_prompt_template": lambda udpt: "--use_default_prompt_template"
    if udpt.lower() == "true"
    else "",
    "enable_tools": lambda et: "" if et.lower() == "true" else "--disable_tools",
}
username = getpass.getuser()
temp_dir = f"/u/{username}/tmp"
os.makedirs(temp_dir, exist_ok=True)
tmp_uuid = uuid.uuid4()

for i, config in enumerate(experiment_configs):
    print(f"Submitting job ({i + 1}/{len(experiment_configs)})...")
    print("Configuration:")
    print(config)
    experiment_name = (
        f"comprehensiveness_{config['variant']}_ng_{args.experiment_version}"
    )
    script = f"""#!/bin/bash
source /u/{username}/.bashrc
conda activate {args.env_name}
python3 {args.repo_path}/src/fact_reasoner/comprehensiveness.py \\
    --experiment_name {experiment_name} \\
"""
    for key, value in config.items():
        if key not in ARG_MAPPING:
            script += f"    --{key} {value} \\\n"
        else:
            line = ARG_MAPPING[key](value)
            if line:
                script += f"    {line} \\\n"
    model_for_filename = regex.sub(r"[^\w\s-_]", "-", config["model"])
    eval_model_for_filename = "" if config["eval_model"] is None else "_" + regex.sub(r"[^\w\s-_]", "-", config["eval_model"])
    log_filename = f"{config['dataset']}_{experiment_name}_{model_for_filename}{eval_model_for_filename}_rt-{config['relevance_threshold']}_dp-{config['use_default_prompt_template']}_tools-{config['enable_tools']}"
    script += f"    >>{args.repo_path}/out/{log_filename}.out 2>&1"
    print("Constructed script:")
    print(script)

    script_path = None
    seconds = 60
    while seconds > 0:
        print(f"Submitting job in {seconds} second(s).", end="\r")
        time.sleep(1)
        seconds -= 1

        if seconds == 40:
            script_path = f"{temp_dir}/run_comp_{tmp_uuid}.sh"
            with open(script_path, "w") as f:
                f.write(script)
            st = os.stat(script_path)
            os.chmod(script_path, st.st_mode | stat.S_IXUSR)
    print("Submitting job...                ")

    assert script_path is not None
    result = subprocess.run(
        ["bsub", "-R", '"rusage[cpu=16, mem=32GB]"', script_path],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print("Job submission complete!")
    print()

print("All jobs submitted!")
