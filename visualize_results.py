"""
Visualization script for anomaly detection results.

This script creates comprehensive visualizations to analyze the anomaly detection output,
including time series plots, score distributions, feature attribution analysis, and more.

Usage:
    python visualize_results.py <output_csv_path>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class AnomalyVisualizer:
    """
    Class for visualizing anomaly detection results.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the visualizer with the output CSV file.
        
        Args:
            csv_path: Path to the anomaly detection output CSV
        """
        self.df = pd.read_csv(csv_path)
        self.timestamp_col = self._find_timestamp_column()
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
        
        # Define time periods
        self.training_start = pd.to_datetime("2004-01-01 00:00:00")
        self.training_end = pd.to_datetime("2004-01-05 23:59:59")
        self.analysis_start = pd.to_datetime("2004-01-01 00:00:00")
        self.analysis_end = pd.to_datetime("2004-01-19 07:59:59")
        
        # Create period masks
        self.training_mask = ((self.df[self.timestamp_col] >= self.training_start) & 
                             (self.df[self.timestamp_col] <= self.training_end))
        self.analysis_mask = ((self.df[self.timestamp_col] >= self.analysis_start) & 
                             (self.df[self.timestamp_col] <= self.analysis_end))
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _find_timestamp_column(self) -> str:
        """Find the timestamp column in the dataframe."""
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp']):
                return col
        raise ValueError("No timestamp column found")
    
    def create_comprehensive_report(self, save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive visualization report.
        
        Args:
            save_path: Optional path to save the report as PDF
        """
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Anomaly Score Time Series
        ax1 = plt.subplot(4, 2, 1)
        self._plot_anomaly_timeline(ax1)
        
        # 2. Score Distribution
        ax2 = plt.subplot(4, 2, 2)
        self._plot_score_distribution(ax2)
        
        # 3. Training vs Non-Training Scores
        ax3 = plt.subplot(4, 2, 3)
        self._plot_training_validation(ax3)
        
        # 4. Top Contributing Features
        ax4 = plt.subplot(4, 2, 4)
        self._plot_feature_importance(ax4)
        
        # 5. Anomaly Severity Heatmap
        ax5 = plt.subplot(4, 2, 5)
        self._plot_severity_heatmap(ax5)
        
        # 6. Feature Attribution Over Time
        ax6 = plt.subplot(4, 2, 6)
        self._plot_feature_attribution_timeline(ax6)
        
        # 7. High Anomaly Events
        ax7 = plt.subplot(4, 2, 7)
        self._plot_high_anomaly_events(ax7)
        
        # 8. Summary Statistics
        ax8 = plt.subplot(4, 2, 8)
        self._plot_summary_stats(ax8)
        
        plt.tight_layout(pad=3.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Report saved to: {save_path}")
        
        plt.show()
    
    def _plot_anomaly_timeline(self, ax) -> None:
        """Plot anomaly scores over time."""
        analysis_data = self.df[self.analysis_mask].copy()
        
        # Plot the main timeline
        ax.plot(analysis_data[self.timestamp_col], analysis_data['abnormality_score'], 
                linewidth=1, alpha=0.7, color='blue', label='Anomaly Score')
        
        # Highlight training period
        training_data = analysis_data[analysis_data[self.timestamp_col] <= self.training_end]
        ax.fill_between(training_data[self.timestamp_col], 0, training_data['abnormality_score'], 
                       alpha=0.3, color='green', label='Training Period')
        
        # Add severity thresholds
        ax.axhline(y=10, color='yellow', linestyle='--', alpha=0.7, label='Normal Threshold')
        ax.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Moderate Anomaly')
        ax.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Significant Anomaly')
        ax.axhline(y=90, color='darkred', linestyle='--', alpha=0.7, label='Severe Anomaly')
        
        ax.set_title('Anomaly Scores Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Abnormality Score')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_score_distribution(self, ax) -> None:
        """Plot distribution of anomaly scores."""
        analysis_data = self.df[self.analysis_mask]
        training_scores = analysis_data[analysis_data[self.timestamp_col] <= self.training_end]['abnormality_score']
        non_training_scores = analysis_data[analysis_data[self.timestamp_col] > self.training_end]['abnormality_score']
        
        # Create histograms
        ax.hist(training_scores, bins=30, alpha=0.7, label='Training Period', color='green', density=True)
        ax.hist(non_training_scores, bins=30, alpha=0.7, label='Post-Training', color='blue', density=True)
        
        # Add vertical lines for means
        ax.axvline(training_scores.mean(), color='green', linestyle='--', 
                  label=f'Training Mean: {training_scores.mean():.2f}')
        ax.axvline(non_training_scores.mean(), color='blue', linestyle='--', 
                  label=f'Post-Training Mean: {non_training_scores.mean():.2f}')
        
        ax.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Abnormality Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_validation(self, ax) -> None:
        """Plot training period validation metrics."""
        analysis_data = self.df[self.analysis_mask]
        training_scores = analysis_data[analysis_data[self.timestamp_col] <= self.training_end]['abnormality_score']
        
        # Calculate statistics
        stats = {
            'Mean': training_scores.mean(),
            'Max': training_scores.max(),
            'Min': training_scores.min(),
            'Std': training_scores.std(),
            '95th Percentile': training_scores.quantile(0.95)
        }
        
        # Create bar plot
        bars = ax.bar(stats.keys(), stats.values(), color=['green', 'red', 'blue', 'orange', 'purple'])
        
        # Add requirement lines
        ax.axhline(y=10, color='red', linestyle='--', label='Mean Requirement (<10)')
        ax.axhline(y=25, color='darkred', linestyle='--', label='Max Requirement (<25)')
        
        # Add value labels on bars
        for bar, value in zip(bars, stats.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Training Period Validation', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_feature_importance(self, ax) -> None:
        """Plot most frequently cited features."""
        feature_cols = [f'top_feature_{i}' for i in range(1, 8)]
        
        # Count feature occurrences
        feature_counts = {}
        for col in feature_cols:
            for feature in self.df[col]:
                if feature and feature != '':
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Get top 15 features
        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if top_features:
            features, counts = zip(*top_features)
            bars = ax.barh(range(len(features)), counts, color='skyblue')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Frequency')
            ax.set_title('Most Frequently Contributing Features', fontsize=14, fontweight='bold')
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                       str(count), ha='left', va='center')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_severity_heatmap(self, ax) -> None:
        """Plot anomaly severity heatmap by day and hour."""
        analysis_data = self.df[self.analysis_mask].copy()
        analysis_data['Day'] = analysis_data[self.timestamp_col].dt.day
        analysis_data['Hour'] = analysis_data[self.timestamp_col].dt.hour
        
        # Create pivot table for heatmap
        heatmap_data = analysis_data.pivot_table(
            values='abnormality_score', 
            index='Day', 
            columns='Hour', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Avg Abnormality Score'})
        ax.set_title('Anomaly Severity Heatmap (Day vs Hour)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Month')
    
    def _plot_feature_attribution_timeline(self, ax) -> None:
        """Plot how feature attributions change over time."""
        analysis_data = self.df[self.analysis_mask].copy()
        
        # Get top 5 most common features
        feature_cols = [f'top_feature_{i}' for i in range(1, 4)]  # Top 3 for clarity
        
        feature_counts = {}
        for col in feature_cols:
            for feature in analysis_data[col]:
                if feature and feature != '':
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        top_features = [f[0] for f in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        # Plot timeline for each top feature
        for i, feature in enumerate(top_features):
            feature_timeline = []
            for _, row in analysis_data.iterrows():
                # Check if feature appears in any of the top 3 positions
                appears = any(row[col] == feature for col in feature_cols)
                feature_timeline.append(1 if appears else 0)
            
            # Smooth the timeline with rolling average
            feature_series = pd.Series(feature_timeline)
            smoothed = feature_series.rolling(window=24, center=True).mean()
            
            ax.plot(analysis_data[self.timestamp_col], smoothed, 
                   label=feature, alpha=0.7, linewidth=2)
        
        ax.set_title('Top Contributing Features Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Contribution Frequency (24h avg)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_high_anomaly_events(self, ax) -> None:
        """Plot timeline of high anomaly events."""
        analysis_data = self.df[self.analysis_mask].copy()
        
        # Define anomaly thresholds
        moderate = analysis_data['abnormality_score'] >= 30
        significant = analysis_data['abnormality_score'] >= 60
        severe = analysis_data['abnormality_score'] >= 90
        
        # Count events by day
        analysis_data['Date'] = analysis_data[self.timestamp_col].dt.date
        daily_stats = analysis_data.groupby('Date').agg({
            'abnormality_score': ['max', 'mean', 'count']
        }).round(2)
        
        # Count anomaly events by day
        daily_anomalies = analysis_data.groupby('Date').apply(lambda x: {
            'moderate': (x['abnormality_score'] >= 30).sum(),
            'significant': (x['abnormality_score'] >= 60).sum(),
            'severe': (x['abnormality_score'] >= 90).sum()
        }).apply(pd.Series)
        
        # Plot stacked bar chart
        dates = daily_anomalies.index
        ax.bar(dates, daily_anomalies['moderate'], label='Moderate (30-59)', color='orange', alpha=0.7)
        ax.bar(dates, daily_anomalies['significant'], bottom=daily_anomalies['moderate'], 
               label='Significant (60-89)', color='red', alpha=0.7)
        ax.bar(dates, daily_anomalies['severe'], 
               bottom=daily_anomalies['moderate'] + daily_anomalies['significant'],
               label='Severe (90-100)', color='darkred', alpha=0.7)
        
        ax.set_title('Daily Anomaly Events by Severity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Events')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_summary_stats(self, ax) -> None:
        """Plot summary statistics table."""
        analysis_data = self.df[self.analysis_mask]
        training_data = analysis_data[analysis_data[self.timestamp_col] <= self.training_end]
        post_training_data = analysis_data[analysis_data[self.timestamp_col] > self.training_end]
        
        # Calculate comprehensive statistics
        stats_data = {
            'Metric': [
                'Total Samples', 'Training Samples', 'Post-Training Samples',
                'Training Mean Score', 'Training Max Score', 'Training Std',
                'Post-Training Mean', 'Post-Training Max', 'Post-Training Std',
                'Moderate Anomalies (30+)', 'Significant Anomalies (60+)', 'Severe Anomalies (90+)',
                'Most Common Feature', 'Validation Status'
            ],
            'Value': [
                len(analysis_data),
                len(training_data),
                len(post_training_data),
                f"{training_data['abnormality_score'].mean():.2f}",
                f"{training_data['abnormality_score'].max():.2f}",
                f"{training_data['abnormality_score'].std():.2f}",
                f"{post_training_data['abnormality_score'].mean():.2f}",
                f"{post_training_data['abnormality_score'].max():.2f}",
                f"{post_training_data['abnormality_score'].std():.2f}",
                len(analysis_data[analysis_data['abnormality_score'] >= 30]),
                len(analysis_data[analysis_data['abnormality_score'] >= 60]),
                len(analysis_data[analysis_data['abnormality_score'] >= 90]),
                self._get_most_common_feature(),
                self._get_validation_status(training_data)
            ]
        }
        
        # Create table
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=[[stat, val] for stat, val in zip(stats_data['Metric'], stats_data['Value'])],
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(stats_data['Metric']) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    def _get_most_common_feature(self) -> str:
        """Get the most commonly contributing feature."""
        feature_cols = [f'top_feature_{i}' for i in range(1, 8)]
        feature_counts = {}
        
        for col in feature_cols:
            for feature in self.df[col]:
                if feature and feature != '':
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        if feature_counts:
            return max(feature_counts.items(), key=lambda x: x[1])[0]
        return "None"
    
    def _get_validation_status(self, training_data: pd.DataFrame) -> str:
        """Check if training period meets validation requirements."""
        mean_score = training_data['abnormality_score'].mean()
        max_score = training_data['abnormality_score'].max()
        
        if mean_score < 10 and max_score < 25:
            return "✅ PASSED"
        else:
            return "❌ FAILED"
    
    def save_detailed_analysis(self, output_path: str) -> None:
        """Save detailed analysis to text file."""
        analysis_data = self.df[self.analysis_mask]
        training_data = analysis_data[analysis_data[self.timestamp_col] <= self.training_end]
        
        with open(output_path, 'w') as f:
            f.write("ANOMALY DETECTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Analysis period samples: {len(analysis_data)}\n")
            f.write(f"Training period samples: {len(training_data)}\n\n")
            
            f.write("TRAINING PERIOD VALIDATION:\n")
            f.write(f"Mean score: {training_data['abnormality_score'].mean():.3f} (requirement: <10)\n")
            f.write(f"Max score: {training_data['abnormality_score'].max():.3f} (requirement: <25)\n")
            f.write(f"Standard deviation: {training_data['abnormality_score'].std():.3f}\n\n")
            
            f.write("ANOMALY STATISTICS:\n")
            f.write(f"Moderate anomalies (30+): {len(analysis_data[analysis_data['abnormality_score'] >= 30])}\n")
            f.write(f"Significant anomalies (60+): {len(analysis_data[analysis_data['abnormality_score'] >= 60])}\n")
            f.write(f"Severe anomalies (90+): {len(analysis_data[analysis_data['abnormality_score'] >= 90])}\n\n")
            
            f.write("TOP 10 HIGHEST ANOMALY EVENTS:\n")
            top_anomalies = analysis_data.nlargest(10, 'abnormality_score')
            for _, row in top_anomalies.iterrows():
                f.write(f"{row[self.timestamp_col]}: {row['abnormality_score']:.2f} "
                       f"(Top feature: {row['top_feature_1']})\n")


def main():
    """Main function to run the visualization."""
    if len(sys.argv) != 2:
        print("Usage: python visualize_results.py <output_csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        print("Loading data and creating visualizations...")
        visualizer = AnomalyVisualizer(csv_path)
        
        # Create comprehensive report
        report_path = csv_path.replace('.csv', '_visualization_report.png')
        visualizer.create_comprehensive_report(save_path=report_path)
        
        # Save detailed analysis
        analysis_path = csv_path.replace('.csv', '_detailed_analysis.txt')
        visualizer.save_detailed_analysis(analysis_path)
        print(f"Detailed analysis saved to: {analysis_path}")
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
