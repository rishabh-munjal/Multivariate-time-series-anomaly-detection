"""
Quick visualization script for anomaly detection results.

This script creates focused visualizations for quick analysis of the anomaly detection output.

Usage:
    python quick_viz.py <output_csv_path>
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime

def create_quick_analysis(csv_path: str):
    """Create quick analysis charts."""
    
    # Load data
    df = pd.read_csv(csv_path)
    timestamp_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp']):
            timestamp_col = col
            break
    
    if not timestamp_col:
        print("Error: No timestamp column found")
        return
    
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Define periods
    training_end = pd.to_datetime("2004-01-05 23:59:59")
    analysis_start = pd.to_datetime("2004-01-01 00:00:00")
    analysis_end = pd.to_datetime("2004-01-19 07:59:59")
    
    # Filter analysis period
    analysis_mask = ((df[timestamp_col] >= analysis_start) & (df[timestamp_col] <= analysis_end))
    analysis_data = df[analysis_mask].copy()
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Anomaly Score Timeline
    ax1.plot(analysis_data[timestamp_col], analysis_data['abnormality_score'], 
             linewidth=1, color='blue', alpha=0.7)
    
    # Highlight training period
    training_data = analysis_data[analysis_data[timestamp_col] <= training_end]
    ax1.fill_between(training_data[timestamp_col], 0, training_data['abnormality_score'], 
                     alpha=0.3, color='green', label='Training Period')
    
    # Add threshold lines
    ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Normal Threshold')
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Anomaly Threshold')
    
    ax1.set_title('Anomaly Scores Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Abnormality Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Score Distribution Comparison
    training_scores = analysis_data[analysis_data[timestamp_col] <= training_end]['abnormality_score']
    post_training_scores = analysis_data[analysis_data[timestamp_col] > training_end]['abnormality_score']
    
    ax2.hist(training_scores, bins=20, alpha=0.7, label='Training Period', color='green', density=True)
    ax2.hist(post_training_scores, bins=20, alpha=0.7, label='Post-Training', color='blue', density=True)
    ax2.axvline(training_scores.mean(), color='green', linestyle='--', label=f'Training Mean: {training_scores.mean():.2f}')
    ax2.axvline(post_training_scores.mean(), color='blue', linestyle='--', label=f'Post-Training Mean: {post_training_scores.mean():.2f}')
    
    ax2.set_title('Score Distribution Comparison')
    ax2.set_xlabel('Abnormality Score')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Top Contributing Features
    feature_cols = [f'top_feature_{i}' for i in range(1, 4)]  # Top 3 for clarity
    feature_counts = {}
    
    for col in feature_cols:
        for feature in analysis_data[col]:
            if feature and feature != '':
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Get top 10 features
    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if top_features:
        features, counts = zip(*top_features)
        bars = ax3.barh(range(len(features)), counts, color='skyblue')
        ax3.set_yticks(range(len(features)))
        ax3.set_yticklabels(features)
        ax3.set_xlabel('Frequency')
        ax3.set_title('Top Contributing Features')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='left', va='center')
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Period Validation
    stats = {
        'Mean': training_scores.mean(),
        'Max': training_scores.max(),
        'Std': training_scores.std(),
        '95th %ile': training_scores.quantile(0.95)
    }
    
    bars = ax4.bar(stats.keys(), stats.values(), color=['green', 'red', 'blue', 'orange'])
    
    # Add requirement lines
    ax4.axhline(y=10, color='red', linestyle='--', label='Mean Req (<10)')
    ax4.axhline(y=25, color='darkred', linestyle='--', label='Max Req (<25)')
    
    # Add value labels
    for bar, value in zip(bars, stats.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_title('Training Period Validation')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = csv_path.replace('.csv', '_quick_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Quick analysis saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("QUICK ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total analysis samples: {len(analysis_data)}")
    print(f"Training samples: {len(training_scores)}")
    print(f"Training period stats:")
    print(f"  - Mean: {training_scores.mean():.3f} (requirement: <10)")
    print(f"  - Max: {training_scores.max():.3f} (requirement: <25)")
    print(f"  - Validation: {'✅ PASSED' if training_scores.mean() < 10 and training_scores.max() < 25 else '❌ FAILED'}")
    print(f"\nAnomaly counts:")
    print(f"  - Moderate (30+): {len(analysis_data[analysis_data['abnormality_score'] >= 30])}")
    print(f"  - Significant (60+): {len(analysis_data[analysis_data['abnormality_score'] >= 60])}")
    print(f"  - Severe (90+): {len(analysis_data[analysis_data['abnormality_score'] >= 90])}")
    
    if top_features:
        print(f"\nTop contributing feature: {top_features[0][0]} ({top_features[0][1]} occurrences)")
    
    plt.show()

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python quick_viz.py <output_csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        create_quick_analysis(csv_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
