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
from Utils.augmentation_generator import create_augmented_dataset, check_augmented_dataset_exists

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

def run_coordinate_descent(cfg):
    """Run coordinate descent: optimize one parameter at a time"""
    print("\n🎯 STARTING COORDINATE DESCENT HYPERPARAMETER OPTIMIZATION")
    
    # Define parameter search order
    param_order = ["optimizer", "learning_rate", "batch_size", "activation", 
                   "dense_layers", "input_size", "epochs"]
    
    # Initialize best configuration with first values
    best_cfg = extract_single_values(cfg)
    best_acc = 0.0
    
    for param_name in param_order:
        if param_name not in cfg:
            continue
            
        param_values = cfg[param_name]
        if not isinstance(param_values, list) or len(param_values) <= 1:
            print(f"⏭️ Skipping {param_name}: only has {len(param_values) if isinstance(param_values, list) else 1} value(s)")
            continue
            
        print(f"\n🔍 OPTIMIZING: {param_name}")
        print(f"📋 Testing {len(param_values)} values: {param_values}")
        
        # Setup output directory for this parameter
        out_base = Path(cfg.get("output_dir", "outputs"))
        results_dir = out_base / "results_csv"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"{param_name}.csv"
        
        rows = []
        param_best_acc = 0.0
        param_best_value = None
        
        for i, value in enumerate(param_values, start=1):
            print(f"\n=== Testing {param_name} {i}/{len(param_values)}: {value} ===")
            
            # Create run config with current best + new parameter value
            run_cfg = copy.deepcopy(best_cfg)
            run_cfg[param_name] = value
            
            # Setup output directory for this run
            combo_name = f"{param_name}_{normalize_value(value)}"
            run_cfg["output_dir"] = str(out_base / combo_name)
            Path(run_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
            
            # Run training
            if cfg.get("use_kfold", False):
                print("🔁 Using K-FOLD training...")
                per_epoch_rows = run_kfold(run_cfg)
            else:
                print("▶️ Using single-run training...")
                per_epoch_rows = train_single(run_cfg)
            
            if not isinstance(per_epoch_rows, list):
                print("⚠️ Expected list of epoch rows but got other. Skipping.")
                continue
            
            # Get best validation accuracy from this run
            if per_epoch_rows:
                run_best_acc = max(row.get('val_acc', 0.0) for row in per_epoch_rows)
                print(f"📊 Best validation accuracy for {param_name}={value}: {run_best_acc:.4f}")
                
                # Update parameter-level best
                if run_best_acc > param_best_acc:
                    param_best_acc = run_best_acc
                    param_best_value = value
                    print(f"🎯 New best for {param_name}: {value} (acc: {run_best_acc:.4f})")
            
            # Add metadata to rows
            for row in per_epoch_rows:
                row_meta = {
                    "parameter": param_name,
                    "param_value": value,
                    "run_id": i,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                # Add current best config
                config_fields = {
                    "optimizer": run_cfg.get("optimizer"),
                    "activation": run_cfg.get("activation"),
                    "batch_size": run_cfg.get("batch_size"),
                    "learning_rate": run_cfg.get("learning_rate"),
                    "dense_layers": run_cfg.get("dense_layers"),
                    "input_size": run_cfg.get("input_size"),
                    "epochs": run_cfg.get("epochs")
                }
                row_final = {**row_meta, **config_fields, **row}
                rows.append(row_final)
        
        # Update best configuration if we found improvement
        if param_best_value is not None and param_best_acc > best_acc:
            best_cfg[param_name] = param_best_value
            best_acc = param_best_acc
            print(f"\n✅ UPDATED BEST CONFIG: {param_name} = {param_best_value}")
            print(f"🏆 New best validation accuracy: {best_acc:.4f}")
        else:
            print(f"\n⚪ NO IMPROVEMENT: keeping {param_name} = {best_cfg[param_name]}")
        
        # Save parameter-specific results
        if rows:
            df = pd.DataFrame(rows)
            # Order columns
            cols = ["parameter", "param_value", "run_id", "timestamp"]
            cols.extend(["optimizer", "activation", "batch_size", "learning_rate", "dense_layers", "input_size", "epochs"])
            cols.extend(["epoch", "train_acc", "val_acc", "test_acc", "train_loss", "val_loss"])
            cols.extend([c for c in df.columns if c not in cols])
            df = df[[c for c in cols if c in df.columns]]
            df.to_csv(csv_path, index=False)
            print(f"📁 Parameter results saved: {csv_path}")
        
        print(f"\n📋 CURRENT BEST CONFIG:")
        for k, v in best_cfg.items():
            if k in param_order:
                print(f"  {k}: {v}")
        print(f"🏆 Best validation accuracy so far: {best_acc:.4f}")
    
    print(f"\n🎉 COORDINATE DESCENT COMPLETED!")
    print(f"🏆 FINAL BEST CONFIGURATION:")
    for k, v in best_cfg.items():
        if k in param_order:
            print(f"  {k}: {v}")
    print(f"🏆 Final best validation accuracy: {best_acc:.4f}")
    
    return best_cfg, best_acc

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

def prepare_data(cfg):
    """Prepare processed and augmented data as needed"""
    
    # Step 1: Ensure processed data exists
    data_proc = Path(cfg.get("data_proc_dir", "Data_processed"))
    if not data_proc.exists() or not any(data_proc.iterdir()):
        print("📌 Processed data not found; running split_dataset...")
        split_dataset(cfg["data_raw_dir"], cfg["data_proc_dir"], seed=cfg.get("seed", 42))
    
    # Step 2: Handle augmented data if enabled
    use_augmented = cfg.get("use_augmented_data", False)
    if use_augmented:
        augmented_dir = cfg.get("data_augmented_dir", "Data_augmented")
        class_names = cfg.get("class_names", ["Jalan Kategori Baik", "Jalan Kurang Baik", "Jalan Rusak"])
        
        # Check if augmented dataset already exists
        if check_augmented_dataset_exists(augmented_dir, class_names):
            print(f"✅ Augmented dataset already exists: {augmented_dir}")
            
            # Show dataset statistics
            augmented_path = Path(augmented_dir)
            for split in ["train", "val", "test"]:
                split_total = 0
                for class_name in class_names:
                    class_dir = augmented_path / split / class_name
                    if class_dir.exists():
                        count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
                        split_total += count
                print(f"  📊 {split}: {split_total} images")
        else:
            print(f"🎨 Creating augmented dataset: {augmented_dir}")
            multiplier = cfg.get("augmentation_multiplier", 5)
            
            success = create_augmented_dataset(
                processed_dir=cfg.get("data_proc_dir", "Data_processed"),
                augmented_dir=augmented_dir,
                multiplier=multiplier,
                class_names=class_names
            )
            
            if not success:
                print("❌ Failed to create augmented dataset, falling back to original data")
                cfg["use_augmented_data"] = False
                return cfg
            
            print(f"✅ Augmented dataset created successfully!")
    
    return cfg

def main():
    cfg = load_config_dict()
    
    # Prepare data (processed + augmented if enabled)
    cfg = prepare_data(cfg)
    
    # Check if multiple parameters have multiple values (coordinate descent mode)
    multi_params = []
    param_keys = ["optimizer", "activation", "batch_size", "learning_rate", 
                  "dense_layers", "input_size", "epochs"]
    
    for key in param_keys:
        if key in cfg and isinstance(cfg[key], list) and len(cfg[key]) > 1:
            multi_params.append(key)
    
    if len(multi_params) > 1:
        print(f"🎯 COORDINATE DESCENT MODE: Found {len(multi_params)} parameters with multiple values")
        print(f"📋 Parameters: {multi_params}")
        run_coordinate_descent(cfg)
    elif len(multi_params) == 1:
        print(f"🔍 SINGLE PARAMETER MODE: Testing {multi_params[0]}")
        run_stage(cfg)
    else:
        print("▶️ SINGLE RUN MODE: All parameters have single values")
        single_cfg = extract_single_values(cfg)
        single_cfg["output_dir"] = cfg.get("output_dir", "outputs") + "/single_run"
        Path(single_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        
        if cfg.get("use_kfold", False):
            print("🔁 Running K-FOLD training...")
            results = run_kfold(single_cfg)
        else:
            print("▶️ Running single training...")
            results = train_single(single_cfg)
        
        print("✅ Training completed!")
        if results:
            best_val_acc = max(row.get('val_acc', 0.0) for row in results)
            print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
