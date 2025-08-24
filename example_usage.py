"""
Example usage script for the anomaly detection system.

This script demonstrates how to use the anomaly detection pipeline
and visualization tools.
"""

import subprocess
import sys
import os

def run_anomaly_detection_example():
    """Run a complete example of the anomaly detection pipeline."""
    
    print("🔍 ANOMALY DETECTION PIPELINE EXAMPLE")
    print("=" * 50)
    
    # Define file paths
    input_file = "81ce1f00-c3f4-4baa-9b57-006fad1875adTEP_Train_Test (1)_with_anomaly_scores.csv"
    output_file = "example_anomaly_output.csv"
    python_exe = "C:/Users/amana/OneDrive/Desktop/Data Science Project/IOT_Multivariate  analysis/.venv/Scripts/python.exe"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    try:
        # Step 1: Run anomaly detection
        print("📊 Step 1: Running anomaly detection...")
        cmd = [python_exe, "anomaly_detection.py", input_file, output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Anomaly detection completed successfully!")
            print(f"Output saved to: {output_file}")
        else:
            print(f" Error in anomaly detection: {result.stderr}")
            return
        
        # Step 2: Create visualizations
        print("\nStep 2: Creating visualizations...")
        
        # Quick visualization
        cmd = [python_exe, "quick_viz.py", output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Quick visualization completed!")
            print(result.stdout)
        else:
            print(f"Error in quick visualization: {result.stderr}")
        
        # Comprehensive visualization
        print("\n📊 Step 3: Creating comprehensive report...")
        cmd = [python_exe, "visualize_results.py", output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Comprehensive visualization completed!")
            print(result.stdout)
        else:
            print(f"Error in comprehensive visualization: {result.stderr}")
        
        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Generated files:")
        print(f"  📄 {output_file} - Anomaly detection results")
        print(f"  🖼️  {output_file.replace('.csv', '_quick_analysis.png')} - Quick analysis chart")
        print(f"  🖼️  {output_file.replace('.csv', '_visualization_report.png')} - Comprehensive report")
        print(f"  📝 {output_file.replace('.csv', '_detailed_analysis.txt')} - Detailed analysis text")
        
    except Exception as e:
        print(f"Unexpected error: {e}")

def show_usage_examples():
    """Show usage examples for each script."""
    
    print("\nUSAGE EXAMPLES")
    print("=" * 50)
    
    print("1. Run anomaly detection:")
    print('   python anomaly_detection.py "input.csv" "output.csv"')
    
    print("\n2. Create quick visualization:")
    print('   python quick_viz.py "output.csv"')
    
    print("\n3. Create comprehensive report:")
    print('   python visualize_results.py "output.csv"')
    
    print("\n4. Run complete pipeline:")
    print('   python example_usage.py')
    
    print("\nREQUIREMENTS:")
    print("- Input CSV must have a timestamp column (containing 'time', 'date', or 'timestamp')")
    print("- Input CSV must have numeric feature columns")
    print("- Training period: 2004-01-01 00:00 to 2004-01-05 23:59")
    print("- Analysis period: 2004-01-01 00:00 to 2004-01-19 07:59")
    
    print("\nOUTPUT VALIDATION:")
    print("- Training period mean score < 10")
    print("- Training period max score < 25")
    print("- Anomaly scores range from 0-100")
    print("- 8 new columns added: abnormality_score + top_feature_1 through top_feature_7")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage_examples()
    else:
        run_anomaly_detection_example()
        show_usage_examples()
