"""
Comparison script for quantum neural network results
Compares multiple CSV result files and creates visualization plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_results(file_pattern="*results*.csv"):
    """
    Load all CSV result files matching the pattern
    """
    csv_files = glob.glob(file_pattern)
    results = {}
    
    for file in csv_files:
        # Extract experiment name from filename
        filename = Path(file).stem
        # Remove timestamp from filename for cleaner labels
        exp_name = filename.replace("_results", "").replace("_", " ")
        if len(exp_name.split()) > 5:  # If name is too long, truncate
            parts = exp_name.split()
            exp_name = " ".join(parts[:4]) + "..."
        
        try:
            df = pd.read_csv(file)
            results[exp_name] = df
            print(f"Loaded: {file} ({len(df)} runs)")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results

def calculate_statistics(results):
    """
    Calculate summary statistics for all experiments
    """
    stats = {}
    
    for exp_name, df in results.items():
        train_acc = df['Train_Accuracy'].values
        test_acc = df['Test_Accuracy'].values
        
        stats[exp_name] = {
            'train_mean': np.mean(train_acc),
            'train_std': np.std(train_acc),
            'train_min': np.min(train_acc),
            'train_max': np.max(train_acc),
            'test_mean': np.mean(test_acc),
            'test_std': np.std(test_acc),
            'test_min': np.min(test_acc),
            'test_max': np.max(test_acc),
            'num_runs': len(df)
        }
    
    return stats

def plot_accuracy_comparison(results):
    """
    Create comparison plots for all experiments
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors for different experiments
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Plot 1: Distribution comparison (histograms)
    ax1 = axes[0, 0]
    for i, (exp_name, df) in enumerate(results.items()):
        ax1.hist(df['Test_Accuracy'] * 100, bins=15, alpha=0.6, 
                label=exp_name, color=colors[i], density=True)
    ax1.set_xlabel('Test Accuracy (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Test Accuracy Distribution Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plots for test accuracy
    ax2 = axes[0, 1]
    test_data = [df['Test_Accuracy'] * 100 for df in results.values()]
    box_plot = ax2.boxplot(test_data, labels=list(results.keys()), patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy Box Plot Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean accuracy with error bars
    ax3 = axes[1, 0]
    exp_names = list(results.keys())
    train_means = [np.mean(df['Train_Accuracy']) * 100 for df in results.values()]
    train_stds = [np.std(df['Train_Accuracy']) * 100 for df in results.values()]
    test_means = [np.mean(df['Test_Accuracy']) * 100 for df in results.values()]
    test_stds = [np.std(df['Test_Accuracy']) * 100 for df in results.values()]
    
    x_pos = np.arange(len(exp_names))
    width = 0.35
    
    ax3.bar(x_pos - width/2, train_means, width, yerr=train_stds, 
            label='Train', alpha=0.8, capsize=5)
    ax3.bar(x_pos + width/2, test_means, width, yerr=test_stds, 
            label='Test', alpha=0.8, capsize=5)
    
    ax3.set_xlabel('Experiment')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Mean Accuracy Comparison (with std dev)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(exp_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning curve comparison (if multiple runs)
    ax4 = axes[1, 1]
    for i, (exp_name, df) in enumerate(results.items()):
        if len(df) > 10:  # Only plot if enough runs
            # Rolling average for smoother curves
            window = max(1, len(df) // 10)
            test_rolling = df['Test_Accuracy'].rolling(window=window, center=True).mean()
            ax4.plot(df['Run'], test_rolling * 100, 
                    label=exp_name, color=colors[i], linewidth=2)
    
    ax4.set_xlabel('Run Number')
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.set_title('Test Accuracy Trends (Rolling Average)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qnn_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_statistical_summary(stats):
    """
    Create a detailed statistical summary plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    exp_names = list(stats.keys())
    test_means = [stats[name]['test_mean'] * 100 for name in exp_names]
    test_stds = [stats[name]['test_std'] * 100 for name in exp_names]
    test_mins = [stats[name]['test_min'] * 100 for name in exp_names]
    test_maxs = [stats[name]['test_max'] * 100 for name in exp_names]
    
    # Plot 1: Mean ± std with min/max range
    x_pos = np.arange(len(exp_names))
    
    # Plot error bars for min/max range
    ax1.errorbar(x_pos, test_means, 
                yerr=[np.array(test_means) - np.array(test_mins),
                      np.array(test_maxs) - np.array(test_means)],
                fmt='o', capsize=5, capthick=2, label='Min/Max Range', alpha=0.6)
    
    # Plot error bars for std dev
    ax1.errorbar(x_pos, test_means, yerr=test_stds, 
                fmt='s', capsize=3, capthick=2, label='Std Dev', color='red')
    
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy Statistics Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(exp_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Train vs Test scatter
    train_means = [stats[name]['train_mean'] * 100 for name in exp_names]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(exp_names)))
    for i, name in enumerate(exp_names):
        ax2.scatter(train_means[i], test_means[i], 
                   s=100, alpha=0.7, color=colors[i], label=name)
        ax2.errorbar(train_means[i], test_means[i], 
                    xerr=stats[name]['train_std'] * 100,
                    yerr=stats[name]['test_std'] * 100,
                    alpha=0.5, color=colors[i])
    
    # Add diagonal line (perfect generalization)
    min_acc = min(min(train_means), min(test_means))
    max_acc = max(max(train_means), max(test_means))
    ax2.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', alpha=0.5, label='Perfect Generalization')
    
    ax2.set_xlabel('Train Accuracy (%)')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Train vs Test Accuracy')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qnn_statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(stats):
    """
    Print a formatted summary table
    """
    print("\n" + "="*80)
    print("QUANTUM NEURAL NETWORK RESULTS COMPARISON")
    print("="*80)
    
    print(f"{'Experiment':<25} {'Runs':<6} {'Test Mean':<12} {'Test Std':<12} {'Train Mean':<12} {'Generalization Gap':<15}")
    print("-"*80)
    
    for name, stat in stats.items():
        gen_gap = (stat['train_mean'] - stat['test_mean']) * 100
        print(f"{name[:24]:<25} {stat['num_runs']:<6} "
              f"{stat['test_mean']*100:<12.2f} {stat['test_std']*100:<12.2f} "
              f"{stat['train_mean']*100:<12.2f} {gen_gap:<15.2f}")
    
    print("-"*80)
    
    # Find best performing experiment
    best_test = max(stats.items(), key=lambda x: x[1]['test_mean'])
    best_stability = min(stats.items(), key=lambda x: x[1]['test_std'])
    best_generalization = min(stats.items(), key=lambda x: x[1]['train_mean'] - x[1]['test_mean'])
    
    print(f"\n🏆 BEST RESULTS:")
    print(f"   Highest Test Accuracy: {best_test[0]} ({best_test[1]['test_mean']*100:.2f}%)")
    print(f"   Most Stable: {best_stability[0]} (std: {best_stability[1]['test_std']*100:.2f}%)")
    print(f"   Best Generalization: {best_generalization[0]} (gap: {(best_generalization[1]['train_mean']-best_generalization[1]['test_mean'])*100:.2f}%)")

def main():
    """
    Main comparison function
    """
    print("🔍 Searching for result CSV files...")
    
    # Load all result files
    results = load_results()
    
    if not results:
        print("❌ No CSV result files found!")
        print("Make sure you have CSV files with 'results' in the filename in the current directory.")
        return
    
    print(f"\n✅ Found {len(results)} experiment(s)")
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Create plots
    print("\n📊 Creating comparison plots...")
    plot_accuracy_comparison(results)
    plot_statistical_summary(stats)
    
    # Print summary
    print_summary_table(stats)
    
    print(f"\n💾 Plots saved:")
    print(f"   - qnn_results_comparison.png")
    print(f"   - qnn_statistical_summary.png")

if __name__ == "__main__":
    main()