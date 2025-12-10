"""
Compare Magikarp (undertrained token detection) results across all checkpoints.
Extracts key metrics from report files and visualizes training progression.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_report(filepath):
    """Extract key metrics from a Magikarp report markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract step number from filename
    filename = os.path.basename(filepath)
    step_match = re.search(r'step_(\d+)', filename)
    metrics['step'] = int(step_match.group(1)) if step_match else 0
    
    # Extract overall distribution (indicator mean and std)
    dist_match = re.search(r'Overall distribution:\s*([\d.]+)\s*\+/-\s*([\d.]+)', content)
    if dist_match:
        metrics['indicator_mean'] = float(dist_match.group(1))
        metrics['indicator_std'] = float(dist_match.group(2))
    
    # Extract number of tested under-trained tokens
    tested_match = re.search(r'Number of tested under-trained tokens:\s*(\d+)', content)
    if tested_match:
        metrics['total_tested'] = int(tested_match.group(1))
    
    # Extract tokens below p=0.01 threshold
    p01_match = re.search(r'(\d+)\s*below p = 0.01 threshold', content)
    if p01_match:
        metrics['below_p01_threshold'] = int(p01_match.group(1))
    
    # Extract tokens below soft indicator threshold
    soft_match = re.search(r'(\d+)\s*below soft indicator threshold', content)
    if soft_match:
        metrics['below_soft_threshold'] = int(soft_match.group(1))
    
    # Extract single byte token info
    single_byte_match = re.search(r'Number of single byte tokens:\s*(\d+),\s*of which\s*(\d+)\s*below', content)
    if single_byte_match:
        metrics['single_byte_total'] = int(single_byte_match.group(1))
        metrics['single_byte_undertrained'] = int(single_byte_match.group(2))
    
    # Extract UTF-fragment token info
    utf_match = re.search(r'non-single-byte UTF-fragment tokens:\s*(\d+),\s*of which\s*(\d+)\s*below', content)
    if utf_match:
        metrics['utf_fragment_total'] = int(utf_match.group(1))
        metrics['utf_fragment_undertrained'] = int(utf_match.group(2))
    
    # Extract verification threshold
    threshold_match = re.search(r'(\d+)\s*entries below threshold of\s*([\d.]+)', content)
    if threshold_match:
        metrics['entries_below_threshold'] = int(threshold_match.group(1))
        metrics['threshold_value'] = float(threshold_match.group(2))
    
    return metrics


def main():
    reports_dir = r"D:\Thesis 2510674\gpt2-engine-main\results\reports"
    
    # Parse all reports
    results = []
    for filename in os.listdir(reports_dir):
        if filename.endswith('.md'):
            filepath = os.path.join(reports_dir, filename)
            metrics = parse_report(filepath)
            results.append(metrics)
    
    # Create DataFrame and sort by step
    df = pd.DataFrame(results)
    df = df.sort_values('step').reset_index(drop=True)
    
    # Display summary table
    print("=" * 80)
    print("MAGIKARP ANALYSIS: COMPARISON ACROSS CHECKPOINTS")
    print("=" * 80)
    print()
    
    # Summary table
    summary_cols = ['step', 'indicator_mean', 'indicator_std', 'below_soft_threshold', 
                    'single_byte_undertrained', 'utf_fragment_undertrained', 'threshold_value']
    available_cols = [c for c in summary_cols if c in df.columns]
    
    print("Summary Table:")
    print("-" * 80)
    print(df[available_cols].to_string(index=False))
    print()
    
    # Calculate changes
    print("=" * 80)
    print("KEY OBSERVATIONS")
    print("=" * 80)
    
    first = df.iloc[0]
    last = df.iloc[-1]
    
    print(f"\nTraining Progress: Step {first['step']:,} → Step {last['step']:,}")
    print(f"  - Tokens processed: ~{first['step'] * 65536 / 1e9:.2f}B → ~{last['step'] * 65536 / 1e9:.2f}B")
    print()
    
    if 'indicator_mean' in df.columns:
        print(f"Indicator Mean (lower = better trained):")
        print(f"  - Start: {first['indicator_mean']:.4f}")
        print(f"  - End:   {last['indicator_mean']:.4f}")
        print(f"  - Change: {last['indicator_mean'] - first['indicator_mean']:.4f} ({(last['indicator_mean'] - first['indicator_mean'])/first['indicator_mean']*100:.1f}%)")
        print()
    
    if 'below_soft_threshold' in df.columns:
        print(f"Undertrained Tokens (below soft threshold):")
        print(f"  - Start: {first['below_soft_threshold']:,}")
        print(f"  - End:   {last['below_soft_threshold']:,}")
        print(f"  - Change: {last['below_soft_threshold'] - first['below_soft_threshold']:+,}")
        print()
    
    if 'single_byte_undertrained' in df.columns:
        print(f"Single-Byte Undertrained Tokens:")
        print(f"  - Start: {first['single_byte_undertrained']}")
        print(f"  - End:   {last['single_byte_undertrained']}")
        print()
    
    if 'threshold_value' in df.columns:
        print(f"Detection Threshold:")
        print(f"  - Start: {first['threshold_value']:.4f}")
        print(f"  - End:   {last['threshold_value']:.4f}")
        print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Magikarp Analysis: Undertrained Token Detection Across Training', fontsize=14, fontweight='bold')
    
    # Plot 1: Indicator Mean over training
    ax1 = axes[0, 0]
    if 'indicator_mean' in df.columns:
        ax1.plot(df['step'], df['indicator_mean'], 'b-o', linewidth=2, markersize=6)
        ax1.fill_between(df['step'], 
                         df['indicator_mean'] - df['indicator_std'],
                         df['indicator_mean'] + df['indicator_std'],
                         alpha=0.3)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Indicator Mean')
        ax1.set_title('E_out Cosine Distance (Lower = Better Trained)')
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Plot 2: Undertrained token count
    ax2 = axes[0, 1]
    if 'below_soft_threshold' in df.columns:
        ax2.plot(df['step'], df['below_soft_threshold'], 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Count')
        ax2.set_title('Tokens Below Soft Indicator Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Plot 3: Single byte and UTF fragment undertrained
    ax3 = axes[1, 0]
    if 'single_byte_undertrained' in df.columns and 'utf_fragment_undertrained' in df.columns:
        ax3.plot(df['step'], df['single_byte_undertrained'], 'g-o', linewidth=2, markersize=6, label='Single Byte')
        ax3.plot(df['step'], df['utf_fragment_undertrained'], 'm-s', linewidth=2, markersize=6, label='UTF Fragment')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Count')
        ax3.set_title('Undertrained Tokens by Type')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Plot 4: Detection threshold evolution
    ax4 = axes[1, 1]
    if 'threshold_value' in df.columns:
        ax4.plot(df['step'], df['threshold_value'], 'c-o', linewidth=2, markersize=6)
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Threshold Value')
        ax4.set_title('Detection Threshold Over Training')
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = r"D:\Thesis 2510674\gpt2-engine-main\results\magikarp_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {output_path}")
    
    # Also save the data to CSV
    csv_path = r"D:\Thesis 2510674\gpt2-engine-main\results\magikarp_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Data saved to: {csv_path}")
    
    plt.show()
    
    return df


if __name__ == "__main__":
    df = main()
