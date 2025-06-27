import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

"""
Task 2.3a: train KNN
"""

class KNNAnalyzer:
    def __init__(self, train_file, val_file=None, test_file=None, output_dir='results/knn'):
        """Initialize KNN Analyzer"""
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.output_dir = output_dir
        
        Path(self.output_dir).mkdir(exist_ok=True)
        
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and prepare data for KNN analysis"""
        print("=== KNN DATA PREPARATION ===")
        
        # Load training data
        self.train_df = pd.read_csv(self.train_file)
        print(f"Training dataset shape: {self.train_df.shape}")
        
        # Encode categorical features
        self.data_type_encoder = LabelEncoder()
        if 'data_type' in self.train_df.columns:
            self.train_df['data_type_encoded'] = self.data_type_encoder.fit_transform(self.train_df['data_type'])
            print(f"Data type encoding: {dict(zip(self.data_type_encoder.classes_, self.data_type_encoder.transform(self.data_type_encoder.classes_)))}")
        
        # Prepare features and target
        columns_to_drop = ['uid', 'data_type', 'pI']
        self.target_col = 'label'

        
        self.X_train = self.train_df.drop(columns_to_drop + [self.target_col], axis=1, errors='ignore')
        self.X_train = self.X_train.select_dtypes(include=[np.number])
        self.y_train = self.train_df[self.target_col]
        
        # Load validation data if provided
        if self.val_file:
            self.val_df = pd.read_csv(self.val_file)
            if 'data_type' in self.val_df.columns:
                self.val_df['data_type_encoded'] = self.data_type_encoder.transform(self.val_df['data_type'])
            
            self.X_val = self.val_df.drop(columns_to_drop + [self.target_col], axis=1, errors='ignore')
            self.X_val = self.X_val.select_dtypes(include=[np.number])
            self.y_val = self.val_df[self.target_col]
            print(f"Validation dataset shape: {self.X_val.shape}")
        else:
            self.X_val, self.y_val = None, None
        
        # Load test data if provided
        if self.test_file:
            self.test_df = pd.read_csv(self.test_file)
            if 'data_type' in self.test_df.columns:
                self.test_df['data_type_encoded'] = self.data_type_encoder.transform(self.test_df['data_type'])
            
            self.X_test = self.test_df.drop(columns_to_drop + [self.target_col], axis=1, errors='ignore')
            self.X_test = self.X_test.select_dtypes(include=[np.number])
            self.y_test = self.test_df[self.target_col]
            print(f"Test dataset shape: {self.X_test.shape}")
        else:
            self.X_test, self.y_test = None, None
        
        self.feature_names = list(self.X_train.columns)
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Target distribution: {self.y_train.value_counts().to_dict()}")
        
        # Use validation set for evaluation if available, otherwise use test set
        if self.X_val is not None:
            self.X_eval, self.y_eval = self.X_val, self.y_val
            self.eval_set_name = "Validation"
        elif self.X_test is not None:
            self.X_eval, self.y_eval = self.X_test, self.y_test
            self.eval_set_name = "Test"
        else:
            # Split training data if no separate evaluation set
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
            )
            self.eval_set_name = "Holdout"
        
        print(f"Using {self.eval_set_name} set for evaluation")
        
        # Initialize scaler for KNN (important for distance-based algorithms)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_eval_scaled = self.scaler.transform(self.X_eval)
        if self.X_test is not None:
            self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data standardized for KNN analysis")

    def test_k_values(self, k_range=range(1, 31)):
        """Test different values of k (number of neighbors)"""
        print(f"\n=== TESTING K VALUES (NUMBER OF NEIGHBORS) ===")
        
        train_scores = []
        eval_scores = []
        cv_scores = []
        
        for k in k_range:
            # Create and train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train_scaled, self.y_train)
            
            # Calculate scores
            train_score = knn.score(self.X_train_scaled, self.y_train)
            eval_score = knn.score(self.X_eval_scaled, self.y_eval)
            cv_score = cross_val_score(knn, self.X_train_scaled, self.y_train, cv=5).mean()
            
            train_scores.append(train_score)
            eval_scores.append(eval_score)
            cv_scores.append(cv_score)
            
            if k <= 10 or k % 5 == 0:  # Print every value up to 10, then every 5th
                print(f"k={k:2d}: Train={train_score:.3f}, {self.eval_set_name}={eval_score:.3f}, CV={cv_score:.3f}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Main performance plot
        plt.subplot(2, 2, 1)
        plt.plot(k_range, train_scores, 'o-', label='Training Accuracy', color='blue', linewidth=2)
        plt.plot(k_range, eval_scores, 'o-', label=f'{self.eval_set_name} Accuracy', color='red', linewidth=2)
        plt.plot(k_range, cv_scores, 'o-', label='CV Accuracy', color='green', linewidth=2)
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Accuracy')
        plt.title('KNN Performance vs Number of Neighbors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Overfitting analysis
        plt.subplot(2, 2, 2)
        overfitting = np.array(train_scores) - np.array(eval_scores)
        plt.plot(k_range, overfitting, 'o-', color='purple', linewidth=2)
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Overfitting (Train - Eval)')
        plt.title('Overfitting Analysis')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Bias-Variance tradeoff visualization
        plt.subplot(2, 2, 3)
        plt.plot(k_range, 1 - np.array(train_scores), 'o-', label='Training Error', color='blue', alpha=0.7)
        plt.plot(k_range, 1 - np.array(eval_scores), 'o-', label=f'{self.eval_set_name} Error', color='red', alpha=0.7)
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Error Rate')
        plt.title('Bias-Variance Tradeoff')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Best k values
        plt.subplot(2, 2, 4)
        best_cv_idx = np.argmax(cv_scores)
        best_eval_idx = np.argmax(eval_scores)
        best_k_cv = k_range[best_cv_idx]
        best_k_eval = k_range[best_eval_idx]
        
        categories = ['Best CV k', f'Best {self.eval_set_name} k', 'Best CV Score', f'Best {self.eval_set_name} Score']
        values = [best_k_cv, best_k_eval, cv_scores[best_cv_idx], eval_scores[best_eval_idx]]
        colors = ['green', 'red', 'lightgreen', 'lightcoral']
        
        bars = plt.bar(categories, values, color=colors, alpha=0.7)
        plt.title('Best Performance Summary')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.0f}' if value > 10 else f'{value:.3f}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/k_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        optimal_k = best_k_cv
        print(f"\nOptimal k (CV): {optimal_k} (Score: {cv_scores[best_cv_idx]:.3f})")
        
        return {
            'k_range': list(k_range),
            'train_scores': train_scores,
            'eval_scores': eval_scores,
            'cv_scores': cv_scores,
            'optimal_k': optimal_k,
            'best_cv_score': cv_scores[best_cv_idx]
        }

    def test_distance_metrics(self, optimal_k=5):
        """Test different distance metrics for KNN"""
        print(f"\n=== TESTING DISTANCE METRICS ===")
        
        # Distance metrics to test
        metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        metric_results = {}
        
        for metric in metrics:
            print(f"Testing {metric} distance...")
            
            if metric == 'minkowski':
                # Test different p values for Minkowski distance
                p_values = [1, 2, 3]
                best_score = 0
                best_p = 2
                
                for p in p_values:
                    knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=metric, p=p)
                    cv_score = cross_val_score(knn, self.X_train_scaled, self.y_train, cv=5).mean()
                    if cv_score > best_score:
                        best_score = cv_score
                        best_p = p
                
                knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=metric, p=best_p)
                metric_results[f'{metric} (p={best_p})'] = best_score
            else:
                knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=metric)
                cv_score = cross_val_score(knn, self.X_train_scaled, self.y_train, cv=5).mean()
                metric_results[metric] = cv_score
            
            print(f"{metric}: {metric_results.get(f'{metric} (p={best_p})' if metric == 'minkowski' else metric):.3f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        metrics_names = list(metric_results.keys())
        scores = list(metric_results.values())
        
        bars = plt.bar(metrics_names, scores, color='skyblue', alpha=0.7, edgecolor='navy')
        plt.ylabel('CV Accuracy')
        plt.title(f'Distance Metrics Comparison (k={optimal_k})')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/distance_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        best_metric = max(metric_results, key=metric_results.get)
        print(f"\nBest distance metric: {best_metric} (Score: {metric_results[best_metric]:.3f})")
        
        return {
            'metric_results': metric_results,
            'best_metric': best_metric,
            'best_metric_score': metric_results[best_metric]
        }

    def test_feature_selection(self, optimal_k=5, best_metric='euclidean'):
        """Test different numbers of features using feature selection"""
        print(f"\n=== TESTING FEATURE SELECTION FOR KNN ===")
        
        max_features = min(20, len(self.feature_names))
        feature_range = range(1, max_features + 1)
        
        # Test SelectKBest
        selectk_scores = []
        selectk_features_list = []
        
        for n_features in feature_range:
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            selectk_features_list.append(selected_features)
            
            # Train KNN with selected features
            knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=best_metric)
            cv_score = cross_val_score(knn, X_train_selected, self.y_train, cv=5).mean()
            selectk_scores.append(cv_score)
            
            if n_features <= 10 or n_features % 5 == 0:
                print(f"SelectKBest {n_features:2d} features: CV={cv_score:.3f}")
        
        # Test RFE (Recursive Feature Elimination)
        print("\nTesting RFE (Recursive Feature Elimination)...")
        rfe_scores = []
        rfe_features_list = []
        
        # Use a simple estimator for RFE (LogisticRegression is faster than KNN)
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(random_state=42, max_iter=1000)
        
        for n_features in feature_range:
            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
            X_train_rfe = rfe.fit_transform(self.X_train_scaled, self.y_train)
            
            # Get selected feature names
            selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if rfe.support_[i]]
            rfe_features_list.append(selected_features)
            
            # Train KNN with RFE selected features
            knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=best_metric)
            cv_score = cross_val_score(knn, X_train_rfe, self.y_train, cv=5).mean()
            rfe_scores.append(cv_score)
            
            if n_features <= 10 or n_features % 5 == 0:
                print(f"RFE {n_features:2d} features: CV={cv_score:.3f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(feature_range, selectk_scores, 'o-', label='SelectKBest', color='blue', linewidth=2)
        plt.plot(feature_range, rfe_scores, 'o-', label='RFE', color='red', linewidth=2)
        plt.xlabel('Number of Features')
        plt.ylabel('CV Accuracy')
        plt.title('Feature Selection Methods Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show improvement over baseline (all features)
        baseline_knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=best_metric)
        baseline_score = cross_val_score(baseline_knn, self.X_train_scaled, self.y_train, cv=5).mean()
        
        plt.subplot(2, 1, 2)
        selectk_improvement = np.array(selectk_scores) - baseline_score
        rfe_improvement = np.array(rfe_scores) - baseline_score
        
        plt.plot(feature_range, selectk_improvement, 'o-', label='SelectKBest Improvement', color='blue', linewidth=2)
        plt.plot(feature_range, rfe_improvement, 'o-', label='RFE Improvement', color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline (All Features)')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy Improvement')
        plt.title('Improvement over Baseline (All Features)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_selection_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal configurations
        best_selectk_idx = np.argmax(selectk_scores)
        best_rfe_idx = np.argmax(rfe_scores)
        
        optimal_selectk_features = feature_range[best_selectk_idx]
        optimal_rfe_features = feature_range[best_rfe_idx]
        
        print(f"\nOptimal SelectKBest: {optimal_selectk_features} features (Score: {selectk_scores[best_selectk_idx]:.3f})")
        print(f"Optimal RFE: {optimal_rfe_features} features (Score: {rfe_scores[best_rfe_idx]:.3f})")
        print(f"Baseline (all features): {baseline_score:.3f}")
        
        # Choose best method
        if selectk_scores[best_selectk_idx] > rfe_scores[best_rfe_idx]:
            best_method = 'SelectKBest'
            best_n_features = optimal_selectk_features
            best_features = selectk_features_list[best_selectk_idx]
            best_score = selectk_scores[best_selectk_idx]
        else:
            best_method = 'RFE'
            best_n_features = optimal_rfe_features
            best_features = rfe_features_list[best_rfe_idx]
            best_score = rfe_scores[best_rfe_idx]
        
        print(f"\nBest feature selection: {best_method} with {best_n_features} features")
        print(f"Selected features: {best_features}")
        
        return {
            'selectk_scores': selectk_scores,
            'rfe_scores': rfe_scores,
            'feature_range': list(feature_range),
            'baseline_score': baseline_score,
            'best_method': best_method,
            'best_n_features': best_n_features,
            'best_features': best_features,
            'best_score': best_score
        }

    def comprehensive_knn_grid_search(self):
        """Comprehensive grid search for KNN hyperparameters"""
        print(f"\n=== COMPREHENSIVE KNN GRID SEARCH ===")
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'chebyshev'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        knn = KNeighborsClassifier()
        
        grid_search = GridSearchCV(
            knn, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Performing comprehensive KNN grid search...")
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        eval_score = best_model.score(self.X_eval_scaled, self.y_eval)
        
        print(f"\nBest KNN parameters: {best_params}")
        print(f"Best CV score: {best_cv_score:.3f}")
        print(f"{self.eval_set_name} accuracy: {eval_score:.3f}")
        
        # Analyze parameter importance
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Plot parameter analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # K values analysis
        k_scores = results_df.groupby('param_n_neighbors')['mean_test_score'].mean()
        axes[0, 0].plot(k_scores.index, k_scores.values, 'o-', color='blue', linewidth=2)
        axes[0, 0].set_xlabel('Number of Neighbors (k)')
        axes[0, 0].set_ylabel('Mean CV Score')
        axes[0, 0].set_title('K Values Performance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weights comparison
        weights_scores = results_df.groupby('param_weights')['mean_test_score'].mean()
        axes[0, 1].bar(weights_scores.index, weights_scores.values, color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[0, 1].set_ylabel('Mean CV Score')
        axes[0, 1].set_title('Weights Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Metrics comparison
        metrics_scores = results_df.groupby('param_metric')['mean_test_score'].mean()
        axes[1, 0].bar(metrics_scores.index, metrics_scores.values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_ylabel('Mean CV Score')
        axes[1, 0].set_title('Distance Metrics Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Algorithm comparison
        algo_scores = results_df.groupby('param_algorithm')['mean_test_score'].mean()
        axes[1, 1].bar(algo_scores.index, algo_scores.values, color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('Mean CV Score')
        axes[1, 1].set_title('Algorithm Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/grid_search_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'eval_score': eval_score,
            'results_df': results_df
        }

    def analyze_knn_decision_boundaries(self, best_model, feature_results=None):
        """Analyze KNN decision boundaries using 2D projections"""
        print(f"\n=== KNN DECISION BOUNDARY ANALYSIS ===")
        
        from sklearn.decomposition import PCA
        
        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        X_train_pca = pca.fit_transform(self.X_train_scaled)
        X_eval_pca = pca.transform(self.X_eval_scaled)
        
        # Train KNN on 2D data
        knn_2d = KNeighborsClassifier(
            n_neighbors=best_model.n_neighbors,
            weights=best_model.weights,
            metric=best_model.metric
        )
        knn_2d.fit(X_train_pca, self.y_train)
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh
        Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(15, 5))
        
        # Decision boundary plot
        plt.subplot(1, 3, 1)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
        
        # Plot training points
        scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=self.y_train, 
                             cmap='RdYlBu', edgecolors='black', alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'KNN Decision Boundary (k={best_model.n_neighbors})')
        plt.colorbar(scatter)
        
        # Feature importance in PCA space
        plt.subplot(1, 3, 2)
        feature_importance_pc1 = np.abs(pca.components_[0])
        feature_importance_pc2 = np.abs(pca.components_[1])
        
        top_features_pc1 = np.argsort(feature_importance_pc1)[-10:]
        plt.barh(range(len(top_features_pc1)), feature_importance_pc1[top_features_pc1])
        plt.yticks(range(len(top_features_pc1)), 
                  [self.feature_names[i].replace('_', ' ')[:15] for i in top_features_pc1])
        plt.xlabel('Absolute Loading')
        plt.title('Top Features Contributing to PC1')
        
        plt.subplot(1, 3, 3)
        top_features_pc2 = np.argsort(feature_importance_pc2)[-10:]
        plt.barh(range(len(top_features_pc2)), feature_importance_pc2[top_features_pc2])
        plt.yticks(range(len(top_features_pc2)), 
                  [self.feature_names[i].replace('_', ' ')[:15] for i in top_features_pc2])
        plt.xlabel('Absolute Loading')
        plt.title('Top Features Contributing to PC2')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/decision_boundary_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate accuracy on 2D projection
        accuracy_2d = knn_2d.score(X_eval_pca, self.y_eval)
        print(f"KNN accuracy on 2D PCA projection: {accuracy_2d:.3f}")
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return {
            'pca_accuracy': accuracy_2d,
            'explained_variance': pca.explained_variance_ratio_.sum(),
            'pca_components': pca.components_
        }

    def final_knn_evaluation(self, best_model):
        """Evaluate final KNN model on test set if available"""
        if self.X_test is not None and self.y_test is not None:
            print(f"\n=== FINAL KNN MODEL EVALUATION ON TEST SET ===")
            
            y_pred = best_model.predict(self.X_test_scaled)
            test_accuracy = accuracy_score(self.y_test, y_pred)
            
            print(f"Final test accuracy: {test_accuracy:.3f}")
            print(f"Model parameters: {best_model.get_params()}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Non-acidic', 'Acidic']))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Non-acidic', 'Acidic'],
                        yticklabels=['Non-acidic', 'Acidic'])
            plt.title('Confusion Matrix - Final KNN Model')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/confusion_matrix_final.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return test_accuracy
        
        return None

    def create_knn_performance_summary(self, k_results, metric_results, feature_results, grid_results):
        """Create comprehensive KNN performance summary"""
        print(f"\n=== KNN PERFORMANCE SUMMARY ===")
        
        # Create summary table
        summary_data = {
            'Method': ['K Optimization', 'Distance Metrics', 'Feature Selection', 'Grid Search'],
            'Best_Parameter': [
                f"k={k_results['optimal_k']}", 
                f"{metric_results['best_metric']}", 
                f"{feature_results['best_method']} ({feature_results['best_n_features']} features)",
                f"Multiple: {grid_results['best_params']}"
            ],
            'CV_Score': [
                k_results['best_cv_score'],
                metric_results['best_metric_score'],
                feature_results['best_score'],
                grid_results['best_cv_score']
            ],
            f'{self.eval_set_name}_Score': [
                k_results['eval_scores'][k_results['optimal_k']-1],  # k-1 because k_range starts from 1
                'N/A',  # Not calculated for metric comparison
                'N/A',  # Not calculated for feature selection
                grid_results['eval_score']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print("\nKNN Performance Summary:")
        print(summary_df)
        
        # Save to CSV
        summary_df.to_csv(f'{self.output_dir}/knn_performance_summary.csv', index=False)
        
        return summary_df

    def run_complete_knn_analysis(self):
        """Run complete KNN analysis"""
        print("="*70)
        print("           K-NEAREST NEIGHBORS ANALYSIS")
        print("="*70)
        
        results = {}
        
        # 1. Test k values
        print("Step 1: Testing k values...")
        k_results = self.test_k_values()
        results['k_analysis'] = k_results
        
        # 2. Test distance metrics
        print("\nStep 2: Testing distance metrics...")
        metric_results = self.test_distance_metrics(k_results['optimal_k'])
        results['metric_analysis'] = metric_results
        
        # 3. Test feature selection
        print("\nStep 3: Testing feature selection...")
        feature_results = self.test_feature_selection(
            k_results['optimal_k'], 
            metric_results['best_metric'].split()[0]  # Get base metric name
        )
        results['feature_analysis'] = feature_results
        
        # 4. Comprehensive grid search
        print("\nStep 4: Comprehensive grid search...")
        grid_results = self.comprehensive_knn_grid_search()
        results['grid_search'] = grid_results
        
        # 5. Decision boundary analysis
        print("\nStep 5: Decision boundary analysis...")
        boundary_results = self.analyze_knn_decision_boundaries(grid_results['best_model'], feature_results)
        results['boundary_analysis'] = boundary_results
        
        # 6. Performance summary
        print("\nStep 6: Creating performance summary...")
        summary = self.create_knn_performance_summary(k_results, metric_results, feature_results, grid_results)
        results['summary'] = summary
        
        # 7. Final evaluation
        print("\nStep 7: Final model evaluation...")
        test_accuracy = self.final_knn_evaluation(grid_results['best_model'])
        if test_accuracy:
            results['test_accuracy'] = test_accuracy
        
        print(f"\n{'='*70}")
        print("K-NEAREST NEIGHBORS ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"All results saved in: {self.output_dir}/")
        
        return results

def main():
    """Main function"""
    # File paths
    train_file = 'data/IPC_classification_features_train.csv'
    val_file = 'data/IPC_classification_features_val.csv'
    test_file = 'data/IPC_classification_features_test.csv'
    
    # Initialize KNN analyzer
    print("Initializing K-Nearest Neighbors Analyzer...")
    
    knn_analyzer = KNNAnalyzer(train_file, val_file, test_file)
    
    # Run complete analysis
    results = knn_analyzer.run_complete_knn_analysis()
    
    return results

if __name__ == "__main__":
    results = main()
