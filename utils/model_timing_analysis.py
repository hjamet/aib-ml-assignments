"""
Script to analyze individual model training times.
Helps identify which models are the slowest and should potentially be removed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
from src.utils.data_loader import load_titanic_data
from src.utils.preprocessing import preprocess_data
from src.utils.model_factory import get_available_models, create_model


def measure_single_model_time(model_name, X, y, problem_type):
    """
    Measure training time for a single model.
    
    Args:
        model_name (str): Name of the model
        X: Features
        y: Target
        problem_type (str): "Classification" or "Regression"
        
    Returns:
        float: Training time in seconds
    """
    # Create model with default hyperparams
    model = create_model(model_name, problem_type, {})
    
    # Measure training time
    start_time = time.perf_counter()
    model.fit(X, y)
    end_time = time.perf_counter()
    
    return end_time - start_time


def analyze_classification_models():
    """Analyze all classification models."""
    print("\n" + "="*70)
    print("CLASSIFICATION MODELS - TIMING ANALYSIS")
    print("="*70)
    
    # Load and prepare data
    df = load_titanic_data()
    selected_features = ['Age', 'Sex', 'Passenger Class', 'Fare', 'Siblings/Spouses', 'Parents/Children', 'Port of Embarkation']
    X, y, _, _, _, _, _, _ = preprocess_data(df, selected_features, "Fill with median", True, 'survived')
    
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Target: survived (classification)")
    
    # Get all models
    available_models = get_available_models("Classification")
    
    # Measure each model
    results = []
    for model_name in available_models.keys():
        print(f"\n  Testing {model_name}...", end=" ")
        try:
            train_time = measure_single_model_time(model_name, X, y, "Classification")
            results.append({
                'Model': model_name,
                'Time (s)': f"{train_time:.3f}",
                'Time_numeric': train_time
            })
            print(f"{train_time:.3f}s")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                'Model': model_name,
                'Time (s)': 'ERROR',
                'Time_numeric': 999
            })
    
    # Sort by time
    results.sort(key=lambda x: x['Time_numeric'])
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS - Sorted by Training Time (fastest to slowest)")
    print("="*70)
    
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'Time_numeric'} for r in results])
    print(df_results.to_string(index=False))
    
    # Summary
    total_time = sum(r['Time_numeric'] for r in results if r['Time_numeric'] != 999)
    print(f"\n  Total time to train all models: {total_time:.3f}s")
    print(f"  Average time per model: {total_time/len(results):.3f}s")
    
    return results


def analyze_regression_models():
    """Analyze all regression models."""
    print("\n" + "="*70)
    print("REGRESSION MODELS - TIMING ANALYSIS")
    print("="*70)
    
    # Load and prepare data
    df = load_titanic_data()
    selected_features = ['Age', 'Sex', 'Passenger Class', 'Siblings/Spouses', 'Parents/Children', 'Port of Embarkation']
    X, y, _, _, _, _, _, _ = preprocess_data(df, selected_features, "Fill with median", True, 'fare')
    
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Target: fare (regression)")
    
    # Get all models
    available_models = get_available_models("Regression")
    
    # Measure each model
    results = []
    for model_name in available_models.keys():
        print(f"\n  Testing {model_name}...", end=" ")
        try:
            train_time = measure_single_model_time(model_name, X, y, "Regression")
            results.append({
                'Model': model_name,
                'Time (s)': f"{train_time:.3f}",
                'Time_numeric': train_time
            })
            print(f"{train_time:.3f}s")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                'Model': model_name,
                'Time (s)': 'ERROR',
                'Time_numeric': 999
            })
    
    # Sort by time
    results.sort(key=lambda x: x['Time_numeric'])
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS - Sorted by Training Time (fastest to slowest)")
    print("="*70)
    
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'Time_numeric'} for r in results])
    print(df_results.to_string(index=False))
    
    # Summary
    total_time = sum(r['Time_numeric'] for r in results if r['Time_numeric'] != 999)
    print(f"\n  Total time to train all models: {total_time:.3f}s")
    print(f"  Average time per model: {total_time/len(results):.3f}s")
    
    return results


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("MODEL TIMING ANALYSIS - Individual Model Performance")
    print("="*70)
    print("\nThis script measures the training time of each model individually")
    print("to help identify which models are slowest and could be removed.")
    
    # Analyze classification
    clf_results = analyze_classification_models()
    
    # Analyze regression
    reg_results = analyze_regression_models()
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("\n📊 Classification - Top 3 Slowest Models:")
    clf_sorted = sorted(clf_results, key=lambda x: x['Time_numeric'], reverse=True)
    for i, r in enumerate(clf_sorted[:3], 1):
        if r['Time_numeric'] != 999:
            print(f"  {i}. {r['Model']}: {r['Time (s)']}s")
    
    print("\n📊 Regression - Top 3 Slowest Models:")
    reg_sorted = sorted(reg_results, key=lambda x: x['Time_numeric'], reverse=True)
    for i, r in enumerate(reg_sorted[:3], 1):
        if r['Time_numeric'] != 999:
            print(f"  {i}. {r['Model']}: {r['Time (s)']}s")
    
    print("\n💡 Consider removing or optimizing the slowest models")
    print("   to improve performance with 100-150 concurrent students.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

