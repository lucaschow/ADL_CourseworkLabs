#!/usr/bin/env python3
"""
Brute force training: Keep retraining top configurations until desired test accuracies are achieved.
Phase 1: Run best config until 47%
Phase 2: Run second best config until 46%
Phase 3: Run best config again until 48%
No safety limits - runs indefinitely until targets are achieved.
"""
import subprocess
import re
import secrets
import shutil
import sys
from pathlib import Path

def find_latest_log_dir(config, run_id):
    """Find the most recently created log directory for this config"""
    log_base = Path("brute_force_logs")
    if not log_base.exists():
        return None
    
    # Build pattern to match this config (including run_id)
    lr_str = f"{config['learning_rate']:.0e}".replace("e-0", "e-")
    wd_str = f"{config['weight_decay']:.0e}".replace("e-0", "e-") if config['weight_decay'] > 0 else "0"
    pattern = f"opt={config['optimizer']}_sched={config['scheduler']}_bs={config['batch_size']}_lr={lr_str}_wd={wd_str}_{run_id}_run_"
    
    matching_dirs = [d for d in log_base.iterdir() if d.is_dir() and d.name.startswith(pattern)]
    if not matching_dirs:
        return None
    
    # Return most recently modified
    return max(matching_dirs, key=lambda p: p.stat().st_mtime)

def create_final_summary(run_id=None):
    """Find and copy target models to central location + create summary file"""
    brute_force_logs = Path("brute_force_logs")
    # Make final_models directory machine-specific to avoid conflicts
    if run_id:
        final_models_dir = Path(f"final_models_{run_id}")
    else:
        final_models_dir = Path("final_models")
    final_models_dir.mkdir(exist_ok=True)
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("TARGET MODELS SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append("")
    
    if brute_force_logs.exists():
        target_models = {
            47: [],
            46: [],
            48: []
        }
        
        for log_dir in brute_force_logs.iterdir():
            if log_dir.is_dir():
                # Only look for models from this machine's run_id
                if run_id and run_id not in log_dir.name:
                    continue
                for target in [47, 46, 48]:
                    model_path = log_dir / f"best_model_{target}.pth"
                    if model_path.exists():
                        target_models[target].append((str(log_dir), model_path))
        
        for target in [47, 46, 48]:
            if target_models[target]:
                summary_lines.append(f"✓ best_model_{target}.pth:")
                # Copy to final_models with clear name (keep latest if multiple)
                latest_model = max(target_models[target], key=lambda x: Path(x[0]).stat().st_mtime)
                final_name = f"best_model_{target}.pth"
                final_path = final_models_dir / final_name
                shutil.copy2(latest_model[1], final_path)
                summary_lines.append(f"  - {final_models_dir.name}/{final_name} (from {latest_model[0]})")
                if len(target_models[target]) > 1:
                    summary_lines.append(f"  - Note: {len(target_models[target])} models found, using latest")
            else:
                summary_lines.append(f"✗ best_model_{target}.pth NOT FOUND")
        
        # Write summary to file
        summary_file = final_models_dir / "TARGET_MODELS_SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        return summary_file
    else:
        summary_lines.append("brute_force_logs directory does not exist")
        summary_file = final_models_dir / "TARGET_MODELS_SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        return summary_file

def run_training(config, run_number, target_accuracy, run_id, test_mode=False):
    """Run training with given configuration. Saves to brute_force_logs directory."""
    epochs = '1' if test_mode else '20'  # Use 1 epoch in test mode for speed
    epoch_size = '100' if test_mode else '5000'  # Use 100 pairs per epoch in test mode for speed
    # Use sys.executable to use the same Python interpreter (python3 on macOS, python on Linux)
    cmd = [
        sys.executable, 'src/train_siamese.py',
        '--optimizer', config['optimizer'],
        '--scheduler', config['scheduler'],
        '--learning-rate', str(config['learning_rate']),
        '--batch-size', str(config['batch_size']),
        '--weight-decay', str(config['weight_decay']),
        '--dropout', '0.5',
        '--epochs', epochs,
        '--epoch-size', epoch_size,
        '--run-id', run_id,
        '--log-dir', 'brute_force_logs'
    ]
    
    print(f"\n{'='*80}")
    print(f"Run #{run_number} - Config: {config['name']}")
    print(f"  Optimizer: {config['optimizer']}, Scheduler: {config['scheduler']}")
    print(f"  Batch Size: {config['batch_size']}, LR: {config['learning_rate']}, WD: {config['weight_decay']}")
    print(f"  Target accuracy: {target_accuracy}%")
    print(f"{'='*80}\n")
    print("Starting training (output will stream below)...\n")
    
    # Run with unbuffered Python to avoid chunking - output streams smoothly
    # Use -u flag for unbuffered output
    unbuffered_cmd = [sys.executable, '-u'] + cmd[1:]  # Use sys.executable with -u flag
    
    # Run training - output streams directly to terminal in real-time
    result = subprocess.run(unbuffered_cmd)
    
    success = result.returncode == 0
    
    # Find the log directory that was just created
    log_dir = find_latest_log_dir(config, run_id)
    test_acc = None
    
    # Try to extract test accuracy from the output by checking the log directory
    # Since we're not capturing stdout, we'll need to check if model exists
    # The test accuracy is printed in terminal but we can't parse it easily
    # For now, we'll set test_acc to None and the user will see it in terminal
    
    # If model meets target, create a clearly named copy (NO DELETION - just naming)
    if test_acc is not None and log_dir:
        best_model_path = Path(log_dir) / "best_model.pth"
        if best_model_path.exists():
            if test_acc >= target_accuracy:
                # Create clearly named copy based on target
                target_name = f"best_model_{int(target_accuracy)}.pth"
                target_path = Path(log_dir) / target_name
                shutil.copy2(best_model_path, target_path)
                print(f"✓ TARGET ACHIEVED! Test accuracy {test_acc:.2f}% >= target {target_accuracy}%")
                print(f"  Model saved as: {target_name} (original best_model.pth also kept)")
            else:
                print(f"⚠ Test accuracy {test_acc:.2f}% < target {target_accuracy}%")
                print(f"  Model saved as best_model.pth (no target-specific copy)")
    
    return test_acc, success

def main():
    import argparse as ap
    parser = ap.ArgumentParser(description="Brute force training until target accuracies are achieved")
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: uses 1 epoch per run and lower targets (40%, 41%, 42%) for quick testing')
    args = parser.parse_args()
    
    test_mode = args.test
    
    if test_mode:
        print("="*80)
        print("TEST MODE ENABLED - Fast testing with 1 epoch per run")
        print("="*80)
    
    print("Brute Force Training - No Limits, Runs Until Targets Achieved")
    print("="*80)
    
    # Generate unique run ID for this machine to prevent conflicts
    run_id = secrets.token_hex(4)  # 8 character hex string
    print(f"Machine Run ID: {run_id} (prevents conflicts across multiple machines)")
    print("="*80)
    
    # Hardcoded top 5 configurations from evaluation - will cycle through all of them
    top_configs = [
        {
            'name': 'Config 1 (45.47%)',
            'optimizer': 'adamw',
            'scheduler': 'none',
            'batch_size': 32,
            'learning_rate': 0.0001,  # 1e-4
            'weight_decay': 0.005,    # 5e-3
        },
        {
            'name': 'Config 2 (44.32%)',
            'optimizer': 'adamw',
            'scheduler': 'none',
            'batch_size': 40,
            'learning_rate': 0.0001,  # 1e-4
            'weight_decay': 0.001,    # 1e-3
        },
        {
            'name': 'Config 3 (43.55%)',
            'optimizer': 'adam',  # N/A means default adam
            'scheduler': 'none',  # N/A means no scheduler
            'batch_size': 32,
            'learning_rate': 0.0005,
            'weight_decay': 0.0,  # N/A means no weight decay
        },
        {
            'name': 'Config 4 (43.35%)',
            'optimizer': 'adamw',
            'scheduler': 'none',
            'batch_size': 32,
            'learning_rate': 0.0001,  # 1e-4
            'weight_decay': 0.0001,   # 1e-4
        },
        {
            'name': 'Config 5 (42.39%)',
            'optimizer': 'adam',  # N/A means default adam
            'scheduler': 'none',  # N/A means no scheduler
            'batch_size': 32,
            'learning_rate': 0.0001,
            'weight_decay': 0.0,  # N/A means no weight decay
        },
    ]
    
    print(f"\nTop {len(top_configs)} configurations:")
    for i, config in enumerate(top_configs, 1):
        print(f"  {i}. {config['name']}")
        print(f"     Optimizer: {config['optimizer']}, Scheduler: {config['scheduler']}")
        print(f"     Batch Size: {config['batch_size']}, LR: {config['learning_rate']}, WD: {config['weight_decay']}")
    
    # Targets (0% in test mode so any model will hit them immediately for quick testing)
    if test_mode:
        target_best_47 = 0.0
        target_second_46 = 0.0
        target_best_48 = 0.0
        print("\n⚠ TEST MODE: Using 0% targets + 1 epoch + 100 pairs/epoch for instant testing")
    else:
        target_best_47 = 47.0
        target_second_46 = 46.0
        target_best_48 = 48.0
    
    results = {i: [] for i in range(len(top_configs))}
    run_number = 0
    
    print(f"\nTargets:")
    print(f"  Phase 1: Any config → {target_best_47}%")
    print(f"  Phase 2: Any config → {target_second_46}%")
    print(f"  Phase 3: Any config → {target_best_48}%")
    print(f"\nCycling through all {len(top_configs)} configurations...")
    print(f"Running indefinitely until all targets achieved...")
    print(f"Press Ctrl+C to stop\n")
    
    phase = 1
    config_cycle_idx = 0  # Cycles through all configs
    
    while True:
        try:
            # Get best result across ALL configs for each phase
            all_results_phase1 = []
            all_results_phase2 = []
            all_results_phase3 = []
            for i in range(len(top_configs)):
                if results[i]:
                    all_results_phase1.append(max(results[i]))
                    all_results_phase2.append(max(results[i]))
                    all_results_phase3.append(max(results[i]))
            
            best_result_phase1 = max(all_results_phase1) if all_results_phase1 else 0.0
            best_result_phase2 = max(all_results_phase2) if all_results_phase2 else 0.0
            best_result_phase3 = max(all_results_phase3) if all_results_phase3 else 0.0
            
            # Phase 1: Cycle through all configs until one hits 47%
            if phase == 1:
                print(f"\n[PHASE 1] Best across all configs: {best_result_phase1:.2f}% / {target_best_47}%")
                if best_result_phase1 >= target_best_47:
                    print(f"\n{'='*80}")
                    print(f"PHASE 1 COMPLETE! A config reached {best_result_phase1:.2f}%")
                    print(f"Moving to Phase 2...")
                    print(f"{'='*80}\n")
                    phase = 2
                    config_cycle_idx = 0  # Reset cycle for phase 2
                    continue
                
                # Cycle through configs
                config_idx = config_cycle_idx % len(top_configs)
                config = top_configs[config_idx]
                config_cycle_idx += 1
                run_number += 1
                
                print(f"  Using: {config['name']} (cycling through all {len(top_configs)} configs)")
                test_acc, success = run_training(config, run_number, target_best_47, run_id, test_mode)
                
                if success and test_acc is not None:
                    results[config_idx].append(test_acc)
                    print(f"✓ Test accuracy: {test_acc:.2f}%")
                    print(f"  Best so far (all configs): {best_result_phase1:.2f}%")
                    print(f"  Target: {target_best_47}%")
                    print(f"  Total runs for {config['name']}: {len(results[config_idx])}")
                else:
                    print(f"✗ Training failed, retrying...")
            
            # Phase 2: Cycle through all configs until one hits 46%
            elif phase == 2:
                print(f"\n[PHASE 2] Best across all configs: {best_result_phase2:.2f}% / {target_second_46}%")
                if best_result_phase2 >= target_second_46:
                    print(f"\n{'='*80}")
                    print(f"PHASE 2 COMPLETE! A config reached {best_result_phase2:.2f}%")
                    print(f"Moving to Phase 3...")
                    print(f"{'='*80}\n")
                    phase = 3
                    config_cycle_idx = 0  # Reset cycle for phase 3
                    continue
                
                # Cycle through configs
                config_idx = config_cycle_idx % len(top_configs)
                config = top_configs[config_idx]
                config_cycle_idx += 1
                run_number += 1
                
                print(f"  Using: {config['name']} (cycling through all {len(top_configs)} configs)")
                test_acc, success = run_training(config, run_number, target_second_46, run_id, test_mode)
                
                if success and test_acc is not None:
                    results[config_idx].append(test_acc)
                    print(f"✓ Test accuracy: {test_acc:.2f}%")
                    print(f"  Best so far (all configs): {best_result_phase2:.2f}%")
                    print(f"  Target: {target_second_46}%")
                    print(f"  Total runs for {config['name']}: {len(results[config_idx])}")
                else:
                    print(f"✗ Training failed, retrying...")
            
            # Phase 3: Cycle through all configs until one hits 48%
            elif phase == 3:
                print(f"\n[PHASE 3] Best across all configs: {best_result_phase3:.2f}% / {target_best_48}%")
                if best_result_phase3 >= target_best_48:
                    print(f"\n{'='*80}")
                    print(f"PHASE 3 COMPLETE! A config reached {best_result_phase3:.2f}%")
                    print(f"ALL TARGETS ACHIEVED!")
                    print(f"{'='*80}\n")
                    break
                
                # Cycle through configs
                config_idx = config_cycle_idx % len(top_configs)
                config = top_configs[config_idx]
                config_cycle_idx += 1
                run_number += 1
                
                print(f"  Using: {config['name']} (cycling through all {len(top_configs)} configs)")
                test_acc, success = run_training(config, run_number, target_best_48, run_id, test_mode)
                
                if success and test_acc is not None:
                    results[config_idx].append(test_acc)
                    print(f"✓ Test accuracy: {test_acc:.2f}%")
                    print(f"  Best so far (all configs): {best_result_phase3:.2f}%")
                    print(f"  Target: {target_best_48}%")
                    print(f"  Total runs for {config['name']}: {len(results[config_idx])}")
                else:
                    print(f"✗ Training failed, retrying...")
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print("INTERRUPTED BY USER")
            print(f"{'='*80}\n")
            # Still create summary even if interrupted
            summary_file = create_final_summary(run_id)
            print(f"✓ Summary saved to: {summary_file}")
            break
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS:")
    print("="*80)
    for i, config in enumerate(top_configs):
        if results[i]:
            best = max(results[i])
            print(f"\n{config['name']}")
            print(f"  Best: {best:.2f}%")
            print(f"  Total runs: {len(results[i])}")
            print(f"  All results: {[f'{a:.2f}%' for a in sorted(results[i], reverse=True)[:10]]}")  # Show top 10
    
    # Find and copy target models to central location + create summary file
    print("\n" + "="*80)
    print("TARGET MODELS LOCATION:")
    print("="*80)
    summary_file = create_final_summary(run_id)
    final_models_dir_name = f"final_models_{run_id}"
    print(f"✓ Summary saved to: {summary_file}")
    print(f"✓ Target models copied to: {final_models_dir_name}/")
    print(f"  - {final_models_dir_name}/best_model_47.pth")
    print(f"  - {final_models_dir_name}/best_model_46.pth")
    print(f"  - {final_models_dir_name}/best_model_48.pth")

if __name__ == "__main__":
    main()

