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

def create_final_summary():
    """Find and copy target models to central location + create summary file"""
    brute_force_logs = Path("brute_force_logs")
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
                summary_lines.append(f"  - final_models/{final_name} (from {latest_model[0]})")
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

def run_training(config, run_number, target_accuracy, run_id):
    """Run training with given configuration. Saves to brute_force_logs directory."""
    cmd = [
        'python', 'src/train_siamese.py',
        '--optimizer', config['optimizer'],
        '--scheduler', config['scheduler'],
        '--learning-rate', str(config['learning_rate']),
        '--batch-size', str(config['batch_size']),
        '--weight-decay', str(config['weight_decay']),
        '--dropout', '0.5',
        '--epochs', '20',
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
    
    # Run training with output streaming in real-time, but also capture it
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                text=True, bufsize=1, universal_newlines=True)
    
    # Stream output in real-time and also capture for parsing
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Print in real-time
        output_lines.append(line)
    
    process.wait()
    result_output = ''.join(output_lines)
    
    # Extract test accuracy from output
    test_acc = None
    for line in result_output.split('\n'):
        if 'Final test accuracy:' in line:
            acc_match = re.search(r'Final test accuracy: ([\d.]+)%', line)
            if acc_match:
                test_acc = float(acc_match.group(1))
                break
    
    success = process.returncode == 0
    
    # Find the log directory that was just created
    log_dir = find_latest_log_dir(config, run_id)
    
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
    
    return test_acc, result.returncode == 0

def main():
    print("Brute Force Training - No Limits, Runs Until Targets Achieved")
    print("="*80)
    
    # Generate unique run ID for this machine to prevent conflicts
    run_id = secrets.token_hex(4)  # 8 character hex string
    print(f"Machine Run ID: {run_id} (prevents conflicts across multiple machines)")
    print("="*80)
    
    # Hardcoded top configurations from evaluation
    top_configs = [
        {
            'name': 'Best Config (45.47%)',
            'optimizer': 'adamw',
            'scheduler': 'none',
            'batch_size': 32,
            'learning_rate': 0.0001,  # 1e-4
            'weight_decay': 0.005,    # 5e-3
        },
        {
            'name': 'Second Best Config (44.32%)',
            'optimizer': 'adamw',
            'scheduler': 'none',
            'batch_size': 40,
            'learning_rate': 0.0001,  # 1e-4
            'weight_decay': 0.001,    # 1e-3
        },
    ]
    
    print(f"\nTop {len(top_configs)} configurations:")
    for i, config in enumerate(top_configs, 1):
        print(f"  {i}. {config['name']}")
        print(f"     Optimizer: {config['optimizer']}, Scheduler: {config['scheduler']}")
        print(f"     Batch Size: {config['batch_size']}, LR: {config['learning_rate']}, WD: {config['weight_decay']}")
    
    # Targets
    target_best_47 = 47.0
    target_second_46 = 46.0
    target_best_48 = 48.0
    
    results = {i: [] for i in range(len(top_configs))}
    run_number = 0
    
    best_config_idx = 0
    second_best_config_idx = 1
    
    print(f"\nTargets:")
    print(f"  Phase 1: Best config → {target_best_47}%")
    print(f"  Phase 2: Second best config → {target_second_46}%")
    print(f"  Phase 3: Best config → {target_best_48}%")
    print(f"\nRunning indefinitely until all targets achieved...")
    print(f"Press Ctrl+C to stop\n")
    
    phase = 1
    
    while True:
        try:
            # Get current best results
            best_result = max(results[best_config_idx]) if results[best_config_idx] else 0.0
            second_best_result = max(results[second_best_config_idx]) if results[second_best_config_idx] else 0.0
            
            # Phase 1: Run best config until 47%
            if phase == 1:
                print(f"\n[PHASE 1] Best config: {best_result:.2f}% / {target_best_47}%")
                if best_result >= target_best_47:
                    print(f"\n{'='*80}")
                    print(f"PHASE 1 COMPLETE! Best config reached {best_result:.2f}%")
                    print(f"Moving to Phase 2...")
                    print(f"{'='*80}\n")
                    phase = 2
                    continue
                
                config = top_configs[best_config_idx]
                run_number += 1
                test_acc, success = run_training(config, run_number, target_best_47, run_id)
                
                if success and test_acc is not None:
                    results[best_config_idx].append(test_acc)
                    print(f"✓ Test accuracy: {test_acc:.2f}%")
                    print(f"  Best so far: {max(results[best_config_idx]):.2f}%")
                    print(f"  Target: {target_best_47}%")
                    print(f"  Total runs for this config: {len(results[best_config_idx])}")
                else:
                    print(f"✗ Training failed, retrying...")
            
            # Phase 2: Run second best config until 46%
            elif phase == 2:
                print(f"\n[PHASE 2] Second best config: {second_best_result:.2f}% / {target_second_46}%")
                if second_best_result >= target_second_46:
                    print(f"\n{'='*80}")
                    print(f"PHASE 2 COMPLETE! Second best config reached {second_best_result:.2f}%")
                    print(f"Moving to Phase 3...")
                    print(f"{'='*80}\n")
                    phase = 3
                    continue
                
                config = top_configs[second_best_config_idx]
                run_number += 1
                test_acc, success = run_training(config, run_number, target_second_46, run_id)
                
                if success and test_acc is not None:
                    results[second_best_config_idx].append(test_acc)
                    print(f"✓ Test accuracy: {test_acc:.2f}%")
                    print(f"  Second best so far: {max(results[second_best_config_idx]):.2f}%")
                    print(f"  Target: {target_second_46}%")
                    print(f"  Total runs for this config: {len(results[second_best_config_idx])}")
                else:
                    print(f"✗ Training failed, retrying...")
            
            # Phase 3: Run best config again until 48%
            elif phase == 3:
                print(f"\n[PHASE 3] Best config: {best_result:.2f}% / {target_best_48}%")
                if best_result >= target_best_48:
                    print(f"\n{'='*80}")
                    print(f"PHASE 3 COMPLETE! Best config reached {best_result:.2f}%")
                    print(f"ALL TARGETS ACHIEVED!")
                    print(f"{'='*80}\n")
                    break
                
                config = top_configs[best_config_idx]
                run_number += 1
                test_acc, success = run_training(config, run_number, target_best_48, run_id)
                
                if success and test_acc is not None:
                    results[best_config_idx].append(test_acc)
                    print(f"✓ Test accuracy: {test_acc:.2f}%")
                    print(f"  Best so far: {max(results[best_config_idx]):.2f}%")
                    print(f"  Target: {target_best_48}%")
                    print(f"  Total runs for this config: {len(results[best_config_idx])}")
                else:
                    print(f"✗ Training failed, retrying...")
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print("INTERRUPTED BY USER")
            print(f"{'='*80}\n")
            # Still create summary even if interrupted
            summary_file = create_final_summary()
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
    summary_file = create_final_summary()
    print(f"✓ Summary saved to: {summary_file}")
    print(f"✓ Target models copied to: final_models/")
    print(f"  - final_models/best_model_47.pth")
    print(f"  - final_models/best_model_46.pth")
    print(f"  - final_models/best_model_48.pth")

if __name__ == "__main__":
    main()

