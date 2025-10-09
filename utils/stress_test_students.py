"""
Stress test script to simulate 50-150 students training all ML models simultaneously.
Tests the system's ability to handle concurrent users training models in parallel.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd

from src.utils.data_loader import load_titanic_data
from src.utils.preprocessing import preprocess_data
from src.utils.model_factory import get_available_models
from src.utils.model_training import train_models


def prepare_classification_data():
    """
    Prepare data for classification task with default settings.

    Returns:
        tuple: (X, y, selected_models)
    """
    # Load data
    df = load_titanic_data()

    # Default settings for classification
    selected_features = [
        "Age",
        "Sex",
        "Passenger Class",
        "Fare",
        "Siblings/Spouses",
        "Parents/Children",
        "Port of Embarkation",
    ]
    missing_age_option = "Fill with median"
    normalize = True
    target_column = "survived"

    # Preprocess
    X, y, feature_names, scaler, _, _, _, _ = preprocess_data(
        df,
        selected_features,
        missing_age_option,
        normalize,
        target_column=target_column,
    )

    # Get all classification models
    available_models = get_available_models("Classification")
    selected_models = list(available_models.keys())

    return X, y, selected_models


def prepare_regression_data():
    """
    Prepare data for regression task with default settings.

    Returns:
        tuple: (X, y, selected_models)
    """
    # Load data
    df = load_titanic_data()

    # Default settings for regression - predict 'fare'
    selected_features = [
        "Age",
        "Sex",
        "Passenger Class",
        "Siblings/Spouses",
        "Parents/Children",
        "Port of Embarkation",
    ]
    missing_age_option = "Fill with median"
    normalize = True
    target_column = "fare"

    # Preprocess
    X, y, feature_names, scaler, _, _, _, _ = preprocess_data(
        df,
        selected_features,
        missing_age_option,
        normalize,
        target_column=target_column,
    )

    # Get all regression models
    available_models = get_available_models("Regression")
    selected_models = list(available_models.keys())

    return X, y, selected_models


def simulate_student_training(
    student_id, X, y, selected_models, problem_type, test_size=20
):
    """
    Simulate a single student training all models.

    Args:
        student_id (int): Student identifier
        X: Features
        y: Target
        selected_models (list): List of model names to train
        problem_type (str): "Classification" or "Regression"
        test_size (int): Test set percentage

    Returns:
        dict: Results containing timing and model information
    """
    start_time = time.perf_counter()

    # Train all models with default hyperparameters
    hyperparams = {}  # Empty dict = use defaults

    models, results, X_train, X_test, y_train, y_test = train_models(
        X, y, selected_models, hyperparams, test_size, problem_type
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    return {
        "student_id": student_id,
        "total_time": total_time,
        "num_models": len(selected_models),
        "models_trained": selected_models,
        "problem_type": problem_type,
    }


def run_concurrent_test(num_students, X, y, selected_models, problem_type):
    """
    Run stress test with N concurrent students.

    Args:
        num_students (int): Number of students to simulate
        X: Features
        y: Target
        selected_models (list): List of model names
        problem_type (str): "Classification" or "Regression"

    Returns:
        dict: Test results and metrics
    """
    print(f"\n{'='*70}")
    print(
        f"Testing {num_students} students training {len(selected_models)} {problem_type} models concurrently"
    )
    print(f"{'='*70}")

    start_time = time.perf_counter()
    results = []

    # Use ThreadPoolExecutor to simulate concurrent students
    with ThreadPoolExecutor(max_workers=num_students) as executor:
        # Submit all student tasks
        futures = {
            executor.submit(
                simulate_student_training, i, X, y, selected_models, problem_type
            ): i
            for i in range(num_students)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            # Progress indicator
            if completed % 10 == 0 or completed == num_students:
                print(f"  Progress: {completed}/{num_students} students completed...")

    end_time = time.perf_counter()
    total_wall_time = end_time - start_time

    # Calculate statistics
    times = [r["total_time"] for r in results]

    stats = {
        "num_students": num_students,
        "problem_type": problem_type,
        "num_models": len(selected_models),
        "total_wall_time": total_wall_time,
        "mean_time_per_student": np.mean(times),
        "median_time_per_student": np.median(times),
        "min_time_per_student": np.min(times),
        "max_time_per_student": np.max(times),
        "std_time_per_student": np.std(times),
        "models": selected_models,
    }

    return stats


def print_test_results(stats):
    """
    Print formatted test results.

    Args:
        stats (dict): Test statistics
    """
    print(f"\n{'='*70}")
    print(f"Results: {stats['num_students']} Students - {stats['problem_type']}")
    print(f"{'='*70}")
    print(f"  Models trained per student: {stats['num_models']}")
    print(f"  Total wall-clock time: {stats['total_wall_time']:.2f}s")
    print(f"\n  Per-Student Training Time:")
    print(f"    Mean:   {stats['mean_time_per_student']:.3f}s")
    print(f"    Median: {stats['median_time_per_student']:.3f}s")
    print(f"    Min:    {stats['min_time_per_student']:.3f}s")
    print(f"    Max:    {stats['max_time_per_student']:.3f}s")
    print(f"    Std:    {stats['std_time_per_student']:.3f}s")
    print(f"\n  Models tested: {', '.join(stats['models'])}")
    print(f"{'='*70}")


def print_comparison_table(all_stats, problem_type):
    """
    Print comparison table across different student counts.

    Args:
        all_stats (list): List of statistics dictionaries
        problem_type (str): "Classification" or "Regression"
    """
    print(f"\n{'='*70}")
    print(f"Comparison Table - {problem_type}")
    print(f"{'='*70}")

    # Create DataFrame for better formatting
    data = []
    for stats in all_stats:
        if stats["problem_type"] == problem_type:
            data.append(
                {
                    "Students": stats["num_students"],
                    "Wall Time (s)": f"{stats['total_wall_time']:.2f}",
                    "Avg/Student (s)": f"{stats['mean_time_per_student']:.3f}",
                    "Median (s)": f"{stats['median_time_per_student']:.3f}",
                    "Min (s)": f"{stats['min_time_per_student']:.3f}",
                    "Max (s)": f"{stats['max_time_per_student']:.3f}",
                    "Models": stats["num_models"],
                }
            )

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print(f"{'='*70}")


def print_summary(classification_stats, regression_stats):
    """
    Print final summary of all tests.

    Args:
        classification_stats (list): Classification test results
        regression_stats (list): Regression test results
    """
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\nClassification Tests:")
    for stats in classification_stats:
        print(
            f"  {stats['num_students']} students: {stats['total_wall_time']:.2f}s total, "
            f"{stats['mean_time_per_student']:.3f}s avg/student"
        )

    print(f"\nRegression Tests:")
    for stats in regression_stats:
        print(
            f"  {stats['num_students']} students: {stats['total_wall_time']:.2f}s total, "
            f"{stats['mean_time_per_student']:.3f}s avg/student"
        )

    # Performance degradation analysis
    if len(classification_stats) >= 2:
        print(f"\nPerformance Degradation (Classification):")
        base_time = classification_stats[0]["mean_time_per_student"]
        for stats in classification_stats[1:]:
            current_time = stats["mean_time_per_student"]
            degradation = ((current_time - base_time) / base_time) * 100
            print(
                f"  {stats['num_students']} vs {classification_stats[0]['num_students']} students: "
                f"{degradation:+.1f}% change in avg time"
            )

    if len(regression_stats) >= 2:
        print(f"\nPerformance Degradation (Regression):")
        base_time = regression_stats[0]["mean_time_per_student"]
        for stats in regression_stats[1:]:
            current_time = stats["mean_time_per_student"]
            degradation = ((current_time - base_time) / base_time) * 100
            print(
                f"  {stats['num_students']} vs {regression_stats[0]['num_students']} students: "
                f"{degradation:+.1f}% change in avg time"
            )

    print(f"\n{'='*70}")


def main():
    """Main stress test execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stress test ML models with concurrent student simulations"
    )
    parser.add_argument(
        "num_students",
        type=int,
        help="Number of students to simulate (e.g., 1, 5, 50, 100, 150)",
    )
    parser.add_argument(
        "--problem-type",
        choices=["classification", "regression", "both"],
        default="both",
        help="Type of problem to test (default: both)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"STRESS TEST: Simulating {args.num_students} Students Training ML Models")
    print("=" * 70)

    # Test configurations
    student_counts = [args.num_students]

    classification_stats = []
    regression_stats = []

    # ==================== CLASSIFICATION TESTS ====================
    if args.problem_type in ["classification", "both"]:
        print("\n" + "=" * 70)
        print("PART 1: CLASSIFICATION TESTS")
        print("=" * 70)

        print("\nPreparing classification data...")
        X_clf, y_clf, models_clf = prepare_classification_data()
        print(f"  Dataset size: {len(X_clf)} samples")
        print(f"  Number of models: {len(models_clf)}")
        print(f"  Models: {', '.join(models_clf)}")

        for num_students in student_counts:
            stats = run_concurrent_test(
                num_students, X_clf, y_clf, models_clf, "Classification"
            )
            print_test_results(stats)
            classification_stats.append(stats)

        if len(classification_stats) > 1:
            print_comparison_table(classification_stats, "Classification")

    # ==================== REGRESSION TESTS ====================
    if args.problem_type in ["regression", "both"]:
        print("\n" + "=" * 70)
        print("PART 2: REGRESSION TESTS")
        print("=" * 70)

        print("\nPreparing regression data...")
        X_reg, y_reg, models_reg = prepare_regression_data()
        print(f"  Dataset size: {len(X_reg)} samples")
        print(f"  Number of models: {len(models_reg)}")
        print(f"  Models: {', '.join(models_reg)}")

        for num_students in student_counts:
            stats = run_concurrent_test(
                num_students, X_reg, y_reg, models_reg, "Regression"
            )
            print_test_results(stats)
            regression_stats.append(stats)

        if len(regression_stats) > 1:
            print_comparison_table(regression_stats, "Regression")

    # ==================== FINAL SUMMARY ====================
    if classification_stats and regression_stats:
        print_summary(classification_stats, regression_stats)

    print("\n✅ Stress test completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
