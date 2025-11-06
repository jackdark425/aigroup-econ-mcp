"""
Test script for all modern computing and machine learning modules
"""
import numpy as np
import pandas as pd
import sys
import traceback

import os

# Add the project root directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def test_random_forest():
    """Test Random Forest implementation"""
    print("Testing Random Forest...")
    try:
        from econometrics.advanced_methods.modern_computing_machine_learning.random_forest import EconRandomForest, random_forest_analysis
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y_reg = np.random.randn(100)
        y_clf = np.random.randint(0, 2, 100)
        
        # Test regression
        rf_reg = EconRandomForest(problem_type='regression', n_estimators=10)
        rf_reg.fit(X, y_reg)
        pred_reg = rf_reg.predict(X)
        imp_reg = rf_reg.feature_importance()
        eval_reg = rf_reg.evaluate(X, y_reg)
        print("  Random Forest Regression: PASSED")
        
        # Test classification
        rf_clf = EconRandomForest(problem_type='classification', n_estimators=10)
        rf_clf.fit(X, y_clf)
        pred_clf = rf_clf.predict(X)
        imp_clf = rf_clf.feature_importance()
        eval_clf = rf_clf.evaluate(X, y_clf)
        print("  Random Forest Classification: PASSED")
        
        # Test analysis function
        rf_analysis_reg = random_forest_analysis(X, y_reg, problem_type='regression')
        rf_analysis_clf = random_forest_analysis(X, y_clf, problem_type='classification')
        print("  Random Forest Analysis Function: PASSED")
        
        return True
    except Exception as e:
        print(f"  Random Forest: FAILED - {e}")
        traceback.print_exc()
        return False


def test_gradient_boosting():
    """Test Gradient Boosting implementation"""
    print("Testing Gradient Boosting...")
    try:
        from econometrics.advanced_methods.modern_computing_machine_learning.gradient_boosting import EconGradientBoosting, gradient_boosting_analysis
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y_reg = np.random.randn(100)
        y_clf = np.random.randint(0, 2, 100)
        
        # Test sklearn regression
        gb_reg = EconGradientBoosting(algorithm='sklearn', problem_type='regression', n_estimators=10)
        gb_reg.fit(X, y_reg)
        pred_reg = gb_reg.predict(X)
        imp_reg = gb_reg.feature_importance()
        eval_reg = gb_reg.evaluate(X, y_reg)
        print("  Gradient Boosting Sklearn Regression: PASSED")
        
        # Test sklearn classification
        gb_clf = EconGradientBoosting(algorithm='sklearn', problem_type='classification', n_estimators=10)
        gb_clf.fit(X, y_clf)
        pred_clf = gb_clf.predict(X)
        imp_clf = gb_clf.feature_importance()
        eval_clf = gb_clf.evaluate(X, y_clf)
        print("  Gradient Boosting Sklearn Classification: PASSED")
        
        # Test analysis function
        gb_analysis_reg = gradient_boosting_analysis(X, y_reg, algorithm='sklearn', problem_type='regression')
        gb_analysis_clf = gradient_boosting_analysis(X, y_clf, algorithm='sklearn', problem_type='classification')
        print("  Gradient Boosting Analysis Function: PASSED")
        
        return True
    except Exception as e:
        print(f"  Gradient Boosting: FAILED - {e}")
        traceback.print_exc()
        return False


def test_support_vector_machine():
    """Test Support Vector Machine implementation"""
    print("Testing Support Vector Machine...")
    try:
        from econometrics.advanced_methods.modern_computing_machine_learning.support_vector_machine import EconSVM, svm_analysis
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y_reg = np.random.randn(100)
        y_clf = np.random.randint(0, 2, 100)
        
        # Test regression
        svm_reg = EconSVM(problem_type='regression', C=0.1)
        svm_reg.fit(X, y_reg)
        pred_reg = svm_reg.predict(X)
        eval_reg = svm_reg.evaluate(X, y_reg)
        print("  SVM Regression: PASSED")
        
        # Test classification
        svm_clf = EconSVM(problem_type='classification', C=0.1)
        svm_clf.fit(X, y_clf)
        pred_clf = svm_clf.predict(X)
        proba_clf = svm_clf.predict_proba(X)
        eval_clf = svm_clf.evaluate(X, y_clf)
        print("  SVM Classification: PASSED")
        
        # Test analysis function
        svm_analysis_reg = svm_analysis(X, y_reg, problem_type='regression', C=0.1)
        svm_analysis_clf = svm_analysis(X, y_clf, problem_type='classification', C=0.1)
        print("  SVM Analysis Function: PASSED")
        
        return True
    except Exception as e:
        print(f"  SVM: FAILED - {e}")
        traceback.print_exc()
        return False


def test_neural_network():
    """Test Neural Network implementation"""
    print("Testing Neural Network...")
    try:
        from econometrics.advanced_methods.modern_computing_machine_learning.neural_network import EconNeuralNetwork, neural_network_analysis
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y_reg = np.random.randn(100)
        y_clf = np.random.randint(0, 3, 100)  # 3 classes for classification
        
        # Test regression
        nn_reg = EconNeuralNetwork(problem_type='regression', hidden_layer_sizes=(10,), max_iter=100)
        nn_reg.fit(X, y_reg)
        pred_reg = nn_reg.predict(X)
        eval_reg = nn_reg.evaluate(X, y_reg)
        print("  Neural Network Regression: PASSED")
        
        # Test classification
        nn_clf = EconNeuralNetwork(problem_type='classification', hidden_layer_sizes=(10,), max_iter=100)
        nn_clf.fit(X, y_clf)
        pred_clf = nn_clf.predict(X)
        proba_clf = nn_clf.predict_proba(X)
        eval_clf = nn_clf.evaluate(X, y_clf)
        print("  Neural Network Classification: PASSED")
        
        # Test analysis function
        nn_analysis_reg = neural_network_analysis(X, y_reg, problem_type='regression', hidden_layer_sizes=(10,))
        nn_analysis_clf = neural_network_analysis(X, y_clf, problem_type='classification', hidden_layer_sizes=(10,))
        print("  Neural Network Analysis Function: PASSED")
        
        return True
    except Exception as e:
        print(f"  Neural Network: FAILED - {e}")
        traceback.print_exc()
        return False


def test_kmeans_clustering():
    """Test K-Means Clustering implementation"""
    print("Testing K-Means Clustering...")
    try:
        from econometrics.advanced_methods.modern_computing_machine_learning.kmeans_clustering import EconKMeans, kmeans_analysis
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        # Test KMeans
        kmeans = EconKMeans(n_clusters=3, n_init=3, max_iter=100)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        centers = kmeans.cluster_centers()
        eval_metrics = kmeans.evaluate(X)
        print("  K-Means Clustering: PASSED")
        
        # Test analysis function
        kmeans_analysis_result = kmeans_analysis(X, n_clusters=3)
        print("  K-Means Analysis Function: PASSED")
        
        return True
    except Exception as e:
        print(f"  K-Means Clustering: FAILED - {e}")
        traceback.print_exc()
        return False


def test_hierarchical_clustering():
    """Test Hierarchical Clustering implementation"""
    print("Testing Hierarchical Clustering...")
    try:
        from econometrics.advanced_methods.modern_computing_machine_learning.hierarchical_clustering import EconHierarchicalClustering, hierarchical_clustering_analysis
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(50, 5)  # Smaller dataset for hierarchical clustering
        
        # Test Hierarchical Clustering
        hc = EconHierarchicalClustering(n_clusters=3)
        hc.fit(X)
        labels = hc.predict()
        eval_metrics = hc.evaluate(X)
        print("  Hierarchical Clustering: PASSED")
        
        # Test analysis function
        hc_analysis_result = hierarchical_clustering_analysis(X, n_clusters=3)
        print("  Hierarchical Clustering Analysis Function: PASSED")
        
        return True
    except Exception as e:
        print(f"  Hierarchical Clustering: FAILED - {e}")
        traceback.print_exc()
        return False


def test_double_ml():
    """Test Double Machine Learning implementation"""
    print("Testing Double Machine Learning...")
    try:
        from econometrics.advanced_methods.modern_computing_machine_learning.double_ml import DoubleML, double_ml_analysis
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.randn(200)
        d = np.random.randn(200)  # Continuous treatment
        
        # Test Double ML
        dml = DoubleML(treatment_type='continuous', n_folds=3)
        dml.fit(X, y, d)
        effect = dml.get_effect()
        se = dml.get_se()
        ci = dml.get_ci()
        pval = dml.get_pval()
        print("  Double Machine Learning: PASSED")
        
        # Test analysis function
        dml_analysis_result = double_ml_analysis(X, y, d)
        print("  Double ML Analysis Function: PASSED")
        
        return True
    except Exception as e:
        print(f"  Double Machine Learning: FAILED - {e}")
        traceback.print_exc()
        return False


def test_causal_forest():
    """Test Causal Forest implementation"""
    print("Testing Causal Forest...")
    try:
        from econometrics.advanced_methods.modern_computing_machine_learning.causal_forest import CausalForest, causal_forest_analysis
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.randn(200)
        w = np.random.randint(0, 2, 200)  # Binary treatment
        
        # Test Causal Forest
        cf = CausalForest(n_estimators=10, min_samples_leaf=5)
        cf.fit(X, y, w)
        pred = cf.predict(X)
        te = cf.estimate_treatment_effect(X, y, w)
        print("  Causal Forest: PASSED")
        
        # Test analysis function
        cf_analysis_result = causal_forest_analysis(X, y, w, n_estimators=10)
        print("  Causal Forest Analysis Function: PASSED")
        
        return True
    except Exception as e:
        print(f"  Causal Forest: FAILED - {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Running tests for all modern computing and machine learning modules...\n")
    
    tests = [
        test_random_forest,
        test_gradient_boosting,
        test_support_vector_machine,
        test_neural_network,
        test_kmeans_clustering,
        test_hierarchical_clustering,
        test_double_ml,
        test_causal_forest
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()