# main.py
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import copy
import itertools
from Trainers.train_single import train_single
from Trainers.train_kfold import run_kfold
from Utils.split_dataset import split_dataset

def load_config_dict():
    cfg_path = Path("configs/config.json")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg

def detect_stage(cfg):
    if isinstance(cfg.get("optimizer", []), list) and len(cfg["optimizer"]) > 1:
        return "optimizer"
    if isinstance(cfg.get("activation", []), list) and len(cfg["activation"]) > 1:
        return "activation"
    if isinstance(cfg.get("batch_size", []), list) and len(cfg["batch_size"]) > 1:
        return "batch"
    if isinstance(cfg.get("learning_rate", []), list) and len(cfg["learning_rate"]) > 1:
        return "lr"
    if isinstance(cfg.get("dense_layers", []), list) and len(cfg["dense_layers"]) > 1:
        return "dense"
    if isinstance(cfg.get("input_size", []), list) and len(cfg["input_size"]) > 1:
        return "inputsize"
    if isinstance(cfg.get("epochs", []), list) and len(cfg["epochs"]) > 1:
        return "epoch"
    return "single"

def get_stage_list(cfg, stage):
    mapping = {
        "optimizer": cfg.get("optimizer", []),
        "activation": cfg.get("activation", []),
        "batch": cfg.get("batch_size", []),
        "lr": cfg.get("learning_rate", []),
        "dense": cfg.get("dense_layers", []),
        "inputsize": cfg.get("input_size", []),
        "epoch": cfg.get("epochs", [])
    }
    return mapping.get(stage, [])

def normalize_value(v):
    # for composing folder names
    return str(v).replace(" ", "").replace("/", "_")

def extract_single_values(cfg):
    """Convert config arrays to single values for actual training."""
    single_cfg = copy.deepcopy(cfg)
    
    # Convert parameter arrays to single values (take first element)
    param_keys = ["optimizer", "activation", "batch_size", "learning_rate", 
                  "dense_layers", "input_size", "epochs"]
    
    for key in param_keys:
        if key in single_cfg and isinstance(single_cfg[key], list):
            single_cfg[key] = single_cfg[key][0]
    
    return single_cfg

def run_stage(cfg):
    stage = detect_stage(cfg)
    print(f"\n🚀 DETECTED STAGE: {stage}\n")
    values = get_stage_list(cfg, stage)
    if not values:
        print("❌ No multi-valued parameter detected. Exiting.")
        return

    out_base = Path(cfg.get("output_dir", "outputs"))
    results_dir = out_base / "results_csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{stage}.csv"

    rows = []
    for i, value in enumerate(values, start=1):
        print(f"\n=== Running {stage} {i}/{len(values)} : {value} ===")
        run_cfg = extract_single_values(cfg)

        # set search variable
        if stage == "optimizer":
            run_cfg["optimizer"] = value
        elif stage == "activation":
            run_cfg["activation"] = value
        elif stage == "batch":
            run_cfg["batch_size"] = value
        elif stage == "lr":
            run_cfg["learning_rate"] = value
        elif stage == "dense":
            run_cfg["dense_layers"] = value
        elif stage == "inputsize":
            run_cfg["input_size"] = value
        elif stage == "epoch":
            run_cfg["epochs"] = value

        # prepare run output dir
        combo_name = f"{stage}_{normalize_value(value)}"
        run_cfg["output_dir"] = str(out_base / combo_name)
        Path(run_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

        # dispatch training (kfold if enabled & applicable)
        if cfg.get("use_kfold", False) and stage in ["inputsize", "epoch"]:
            print("🔁 Using K-FOLD training for this run...")
            per_epoch_rows = run_kfold(run_cfg)  # returns list of rows
        else:
            print("▶️ Using single-run training for this run...")
            per_epoch_rows = train_single(run_cfg)  # returns list of rows

        # ensure we got list
        if not isinstance(per_epoch_rows, list):
            print("⚠️ Expected list of epoch rows but got other. Skipping.")
            continue

        # attach stage/value metadata and append to master rows
        for row in per_epoch_rows:
            row_meta = {
                "stage": stage,
                "value": value,
                "run_id": i,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            # Put parameters at front: ensure keys exist
            # gather param fields
            param_fields = {
                "optimizer": run_cfg.get("optimizer"),
                "activation": run_cfg.get("activation"),
                "batch_size": run_cfg.get("batch_size"),
                "learning_rate": run_cfg.get("learning_rate"),
                "dense_layers": run_cfg.get("dense_layers"),
                "input_size": run_cfg.get("input_size"),
            }
            row_final = {**row_meta, **param_fields, **row}
            rows.append(row_final)

        print(f"✅ Finished {stage}={value}")

    # Save CSV with columns ordered: params -> epoch -> accuracies -> losses -> time
    if rows:
        df = pd.DataFrame(rows)
        # desired column order
        cols = []
        # meta + params
        meta_params = ["stage", "value", "run_id", "timestamp",
                       "optimizer", "activation", "batch_size", "learning_rate", "dense_layers", "input_size"]
        for c in meta_params:
            if c in df.columns:
                cols.append(c)
        # epoch + accuracies
        for c in ["epoch", "train_acc", "val_acc", "test_acc"]:
            if c in df.columns:
                cols.append(c)
        # losses
        for c in ["train_loss", "val_loss"]:
            if c in df.columns:
                cols.append(c)
        # time
        for c in ["epoch_time", "total_time_s"]:
            if c in df.columns:
                cols.append(c)
        # add any remaining columns
        for c in df.columns:
            if c not in cols:
                cols.append(c)
        df = df[cols]
        df.to_csv(csv_path, index=False)
        print(f"\n📁 Stage results saved: {csv_path}")
    else:
        print("❌ No rows were generated for this stage.")

def main():
    cfg = load_config_dict()
    data_proc = Path(cfg.get("data_proc_dir", "data_processed"))
    if not data_proc.exists() or not any(data_proc.iterdir()):
        print("📌 Processed data not found; running split_dataset...")
        split_dataset(cfg["data_raw_dir"], cfg["data_proc_dir"], seed=cfg.get("seed", 42))
    run_stage(cfg)

if __name__ == "__main__":
    main()
