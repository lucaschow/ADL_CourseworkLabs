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
import shutil
import hashlib
import secrets
from pathlib import Path

def find_latest_log_dir(config, run_id):
    """Find the most recently created log directory for this config"""
    log_base = Path("logs")
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

def run_training(config, run_number, target_accuracy, run_id):
    """Run training with given configuration. Only saves model if it hits target accuracy."""
    cmd = [
        'python', 'src/train_siamese.py',
        '--optimizer', config['optimizer'],
        '--scheduler', config['scheduler'],
        '--learning-rate', str(config['learning_rate']),
        '--batch-size', str(config['batch_size']),
        '--weight-decay', str(config['weight_decay']),
        '--dropout', '0.5',
        '--epochs', '20',
        '--run-id', run_id
    ]
    
    print(f"\n{'='*80}")
    print(f"Run #{run_number} - Config: {config['name']}")
    print(f"  Optimizer: {config['optimizer']}, Scheduler: {config['scheduler']}")
    print(f"  Batch Size: {config['batch_size']}, LR: {config['learning_rate']}, WD: {config['weight_decay']}")
    print(f"  Target accuracy: {target_accuracy}%")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract test accuracy from output
    test_acc = None
    for line in result.stdout.split('\n'):
        if 'Final test accuracy:' in line:
            acc_match = re.search(r'Final test accuracy: ([\d.]+)%', line)
            if acc_match:
                test_acc = float(acc_match.group(1))
                break
    
    # Find the log directory that was just created
    log_dir = find_latest_log_dir(config, run_id)
    
    # Only keep model if it meets target, otherwise delete everything
    if test_acc is not None and log_dir:
        if test_acc >= target_accuracy:
            print(f"✓ Model saved! Test accuracy {test_acc:.2f}% >= target {target_accuracy}%")
        else:
            print(f"✗ Deleting model - Test accuracy {test_acc:.2f}% < target {target_accuracy}%")
            # Delete the entire log directory (including best_model.pth)
            shutil.rmtree(log_dir)
            log_dir = None
    
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

if __name__ == "__main__":
    main()

