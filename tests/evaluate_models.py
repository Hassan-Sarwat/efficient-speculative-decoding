import argparse
import subprocess
import os
import csv
import glob
import json

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=False) # check=False so we don't crash entire script on one model fail

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--data-path", help="Path to test data. If not provided, defaults to HF gsm8k for easy, and looks for local files for others.")
    args = parser.parse_args()

    scenario = args.scenario
    base_model = args.base_model
    
    # Determine Data Path
    data_arg = ""
    if args.data_path:
        data_arg = f"--data-path {args.data_path}"
    else:
        # Default to pre-prepared test files
        test_file = f"data/tests/{scenario}_test.jsonl"
        if os.path.exists(test_file):
            data_arg = f"--data-path {test_file}"
        else:
            print(f"Warning: Pre-prepared test file '{test_file}' not found. Please run 'scripts/prepare_test_data.py' first.")

    # Define Configurations
    # 1. Base Untrained
    # 2. Untrained Speculative
    # 3. Trained CoT
    # 4. Trained CoT Speculative
    # 5. Trained CoD
    # 6. Trained CoD + Speculative
    
    configs = [
        {
            "name": "base_untrained",
            "target": base_model,
            "draft": None,
            "spec": False
        },
        {
            "name": "untrained_speculative",
            "target": base_model,
            "draft": f"models/draft_untrained_{scenario}",
            "spec": True
        },
        {
            "name": "trained_cot",
            "target": f"models/target_cot_{scenario}",
            "draft": None,
            "spec": False
        },
        {
            "name": "trained_cot_speculative",
            "target": f"models/target_cot_{scenario}",
            "draft": f"models/draft_cot_{scenario}",
            "spec": True
        },
        {
            "name": "trained_cod",
            "target": f"models/target_cod_{scenario}",
            "draft": None,
            "spec": False
        },
        {
            "name": "trained_cod_speculative",
            "target": f"models/target_cod_{scenario}",
            "draft": f"models/draft_cod_{scenario}", # Assuming this exists or will exist?
            # User said: "trained cod model with speculative decoding". 
            # Usually CoD is draft-only or target-only? 
            # If CoD is the target, we might use a distilled CoD draft?
            # I'll assume standard naming convention.
            "spec": True
        }
    ]

    # Run Benchmarks
    for conf in configs:
        target_path = conf['target']
        if os.path.isdir(target_path) or "models/" in target_path:
            # It's an adapter path
            target_args = f"--target-base-model {base_model} --target-adapter {target_path}"
        else:
            # It's a base model path (like huggingface hub id)
            target_args = f"--target-base-model {target_path}"
            
        cmd = f"python tests/benchmark.py --scenario {scenario} {target_args} --run-name {conf['name']} {data_arg}"
        
        if conf['spec']:
            draft_path = conf['draft']
            # Assume draft is always an adapter if it matches models/ pattern, else base
            # Hardcoded draft base for now as per project config
            draft_base = "Qwen/Qwen3-0.6B" 
            
            if draft_path and (os.path.isdir(draft_path) or "models/" in draft_path):
                 # benchmark.py exposes --merged-draft-model (the merged adapter path)
                 cmd += f" --use-speculative --draft-base-model {draft_base} --merged-draft-model {draft_path}"
            else:
                 cmd += f" --use-speculative --draft-base-model {draft_base}"

        print(f"\nLaunching Config: {conf['name']}")
        run_cmd(cmd)

    # Aggregate Results
    #
    # benchmark.py writes one JSON per pass at:
    #     outputs/metrics_{run_name}_{baseline|speculative}.json
    # The schema is the BenchmarkMetrics dataclass — same keys for every pass,
    # so we can stack them into a single CSV with one row per pass.
    print("\nAggregating Results...")
    metrics_files = sorted(glob.glob("outputs/metrics_*.json"))
    valid_names = {c['name'] for c in configs}

    agg_path = f"outputs/comparison_{scenario}.csv"
    all_rows = []
    headers = None

    for mf in metrics_files:
        fname = os.path.basename(mf)
        run_id = fname.removeprefix("metrics_").removesuffix(".json")
        # Match against pass names like "{config}_baseline" / "{config}_speculative".
        # Skip the unified per-config aggregate file (metrics_{config}.json with
        # 'baseline'/'speculative' nested keys) — its schema differs.
        config_name = run_id
        for suffix in ("_baseline", "_speculative"):
            if run_id.endswith(suffix):
                config_name = run_id[: -len(suffix)]
                break
        if config_name not in valid_names:
            continue

        try:
            with open(mf, "r") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {mf}: {e}")
            continue

        # Skip the nested unified file (its top-level keys are 'baseline'/'speculative')
        if "experiment_name" not in payload:
            continue

        if headers is None:
            headers = ["config", "run_id"] + list(payload.keys())
        all_rows.append([config_name, run_id] + [payload.get(k) for k in headers[2:]])

    if all_rows:
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(all_rows)
        print(f"Comparison saved to {agg_path} ({len(all_rows)} rows)")
    else:
        print("No metrics found to aggregate.")

if __name__ == "__main__":
    main()
