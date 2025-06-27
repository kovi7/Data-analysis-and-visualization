import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


"""
Task 2.3b: train SVM
"""

class SVMAnalyzer:
    def __init__(self, train_file, val_file=None, test_file=None, output_dir='results/svm'):
        """Initialize SVM Analyzer"""
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.output_dir = output_dir
        
        Path(self.output_dir).mkdir(exist_ok=True)
        
        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        """Load and prepare data for SVM analysis"""
        print("=== SVM DATA PREPARATION ===")
        
        # Load data
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
            self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
            )
            self.eval_set_name = "Holdout"
        
        print(f"Using {self.eval_set_name} set for evaluation")
        
        # Standardization is crucial for SVM
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_eval_scaled = self.scaler.transform(self.X_eval)
        if self.X_test is not None:
            self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data standardized for SVM analysis")

    def test_c_parameter(self, gamma='scale'):
        """Test different C (regularization) parameter values"""
        print(f"\n=== TESTING C PARAMETER (REGULARIZATION) ===")
        
        c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        train_scores = []
        eval_scores = []
        cv_scores = []
        
        for c in c_range:
            svm = SVC(C=c, gamma=gamma, kernel='rbf', random_state=42)
            svm.fit(self.X_train_scaled, self.y_train)
            
            train_score = svm.score(self.X_train_scaled, self.y_train)
            eval_score = svm.score(self.X_eval_scaled, self.y_eval)
            cv_score = cross_val_score(svm, self.X_train_scaled, self.y_train, cv=5).mean()
            
            train_scores.append(train_score)
            eval_scores.append(eval_score)
            cv_scores.append(cv_score)
            
            print(f"C={c:6.3f}: Train={train_score:.3f}, {self.eval_set_name}={eval_score:.3f}, CV={cv_score:.3f}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Main performance plot
        plt.subplot(2, 2, 1)
        plt.semilogx(c_range, train_scores, 'o-', label='Training Accuracy', color='blue', linewidth=2)
        plt.semilogx(c_range, eval_scores, 'o-', label=f'{self.eval_set_name} Accuracy', color='red', linewidth=2)
        plt.semilogx(c_range, cv_scores, 'o-', label='CV Accuracy', color='green', linewidth=2)
        plt.xlabel('C (Regularization Parameter)')
        plt.ylabel('Accuracy')
        plt.title('SVM Performance vs C Parameter')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Overfitting analysis
        plt.subplot(2, 2, 2)
        overfitting = np.array(train_scores) - np.array(eval_scores)
        plt.semilogx(c_range, overfitting, 'o-', color='purple', linewidth=2)
        plt.xlabel('C (Regularization Parameter)')
        plt.ylabel('Overfitting (Train - Eval)')
        plt.title('Overfitting vs C Parameter')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Bias-Variance tradeoff
        plt.subplot(2, 2, 3)
        plt.semilogx(c_range, 1 - np.array(train_scores), 'o-', label='Training Error', color='blue', alpha=0.7)
        plt.semilogx(c_range, 1 - np.array(eval_scores), 'o-', label=f'{self.eval_set_name} Error', color='red', alpha=0.7)
        plt.xlabel('C (Regularization Parameter)')
        plt.ylabel('Error Rate')
        plt.title('Bias-Variance Tradeoff')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Best C value
        plt.subplot(2, 2, 4)
        best_cv_idx = np.argmax(cv_scores)
        best_c = c_range[best_cv_idx]
        
        plt.bar(['Best C', 'Best CV Score'], [best_c, cv_scores[best_cv_idx]], 
                color=['orange', 'green'], alpha=0.7)
        plt.title('Best C Parameter')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/c_parameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        optimal_c = best_c
        print(f"\nOptimal C: {optimal_c} (CV Score: {cv_scores[best_cv_idx]:.3f})")
        
        return {
            'c_range': c_range,
            'train_scores': train_scores,
            'eval_scores': eval_scores,
            'cv_scores': cv_scores,
            'optimal_c': optimal_c,
            'best_cv_score': cv_scores[best_cv_idx]
        }

    def test_gamma_parameter(self, optimal_c=1):
        """Test different gamma parameter values"""
        print(f"\n=== TESTING GAMMA PARAMETER ===")
        
        gamma_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        gamma_scores = []
        
        for gamma in gamma_range:
            svm = SVC(C=optimal_c, gamma=gamma, kernel='rbf', random_state=42)
            cv_score = cross_val_score(svm, self.X_train_scaled, self.y_train, cv=5).mean()
            gamma_scores.append(cv_score)
            print(f"Gamma={gamma:6.4f}: CV={cv_score:.3f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.semilogx(gamma_range, gamma_scores, 'o-', color='red', linewidth=2, markersize=8)
        plt.xlabel('Gamma Parameter')
        plt.ylabel('CV Accuracy')
        plt.title(f'SVM Performance vs Gamma Parameter (C={optimal_c})')
        plt.grid(True, alpha=0.3)
        
        # Mark best gamma
        best_gamma_idx = np.argmax(gamma_scores)
        best_gamma = gamma_range[best_gamma_idx]
        plt.axvline(x=best_gamma, color='green', linestyle='--', alpha=0.7, 
                   label=f'Best Gamma: {best_gamma}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/gamma_parameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nOptimal Gamma: {best_gamma} (CV Score: {gamma_scores[best_gamma_idx]:.3f})")
        
        return {
            'gamma_range': gamma_range,
            'gamma_scores': gamma_scores,
            'optimal_gamma': best_gamma,
            'best_gamma_score': gamma_scores[best_gamma_idx]
        }

    def test_kernel_comparison(self, optimal_c=1):
        """Compare different SVM kernels"""
        print(f"\n=== TESTING DIFFERENT KERNELS ===")
        
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        kernel_results = {}
        
        for kernel in kernels:
            print(f"Testing {kernel} kernel...")
            
            if kernel == 'poly':
                # Test different degrees for polynomial kernel
                degrees = [2, 3, 4]
                best_score = 0
                best_degree = 3
                
                for degree in degrees:
                    svm = SVC(C=optimal_c, kernel=kernel, degree=degree, random_state=42)
                    cv_score = cross_val_score(svm, self.X_train_scaled, self.y_train, cv=5).mean()
                    if cv_score > best_score:
                        best_score = cv_score
                        best_degree = degree
                
                kernel_results[f'{kernel} (degree={best_degree})'] = best_score
            else:
                svm = SVC(C=optimal_c, kernel=kernel, random_state=42)
                cv_score = cross_val_score(svm, self.X_train_scaled, self.y_train, cv=5).mean()
                kernel_results[kernel] = cv_score
            
            print(f"{kernel}: {kernel_results.get(f'{kernel} (degree={best_degree})' if kernel == 'poly' else kernel):.3f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        kernel_names = list(kernel_results.keys())
        scores = list(kernel_results.values())
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
        bars = plt.bar(kernel_names, scores, color=colors, alpha=0.7, edgecolor='navy')
        plt.ylabel('CV Accuracy')
        plt.title(f'Kernel Comparison (C={optimal_c})')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/kernel_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        best_kernel = max(kernel_results, key=kernel_results.get)
        print(f"\nBest kernel: {best_kernel} (Score: {kernel_results[best_kernel]:.3f})")
        
        return {
            'kernel_results': kernel_results,
            'best_kernel': best_kernel,
            'best_kernel_score': kernel_results[best_kernel]
        }

    def comprehensive_grid_search_rbf(self):
        """Comprehensive grid search for RBF SVM"""
        print(f"\n=== COMPREHENSIVE GRID SEARCH FOR RBF SVM ===")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
            'kernel': ['rbf']
        }
        
        svm = SVC(random_state=42)
        
        grid_search = GridSearchCV(
            svm, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Performing comprehensive SVM grid search...")
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        eval_score = best_model.score(self.X_eval_scaled, self.y_eval)
        
        print(f"\nBest SVM parameters: {best_params}")
        print(f"Best CV score: {best_cv_score:.3f}")
        print(f"{self.eval_set_name} accuracy: {eval_score:.3f}")
        
        # Analyze grid search results
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Create heatmap of C vs Gamma
        pivot_table = results_df.pivot_table(
            values='mean_test_score', 
            index='param_gamma', 
            columns='param_C'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', 
                   cbar_kws={'label': 'CV Accuracy'})
        plt.title('SVM Grid Search Results: C vs Gamma')
        plt.xlabel('C Parameter')
        plt.ylabel('Gamma Parameter')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/grid_search_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'eval_score': eval_score,
            'results_df': results_df
        }

    def analyze_svm_decision_boundary(self, best_model):
        """Analyze SVM decision boundary using 2D projection"""
        print(f"\n=== SVM DECISION BOUNDARY ANALYSIS ===")
        
        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        X_train_pca = pca.fit_transform(self.X_train_scaled)
        X_eval_pca = pca.transform(self.X_eval_scaled)
        
        # Train SVM on 2D data
        svm_2d = SVC(
            C=best_model.C,
            gamma=best_model.gamma,
            kernel=best_model.kernel,
            random_state=42
        )
        svm_2d.fit(X_train_pca, self.y_train)
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh
        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(15, 5))
        
        # Decision boundary plot
        plt.subplot(1, 3, 1)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
        
        # Plot training points
        scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=self.y_train, 
                             cmap='RdYlBu', edgecolors='black', alpha=0.7)
        
        # Plot support vectors
        support_vectors = svm_2d.support_vectors_
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                   s=100, facecolors='none', edgecolors='red', linewidth=2, 
                   label=f'Support Vectors ({len(support_vectors)})')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'SVM Decision Boundary (C={best_model.C}, γ={best_model.gamma})')
        plt.legend()
        plt.colorbar(scatter)
        
        # Support vector analysis
        plt.subplot(1, 3, 2)
        n_support = svm_2d.n_support_
        classes = ['Non-acidic', 'Acidic']
        colors = ['lightblue', 'lightcoral']
        
        bars = plt.bar(classes, n_support, color=colors, alpha=0.7, edgecolor='navy')
        plt.ylabel('Number of Support Vectors')
        plt.title('Support Vectors by Class')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, n_support):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom')
        
        # Model complexity analysis
        plt.subplot(1, 3, 3)
        total_support_vectors = len(support_vectors)
        total_training_points = len(X_train_pca)
        support_vector_ratio = total_support_vectors / total_training_points
        
        complexity_metrics = ['Total Training Points', 'Support Vectors', 'SV Ratio (%)']
        values = [total_training_points, total_support_vectors, support_vector_ratio * 100]
        colors = ['lightgreen', 'orange', 'purple']
        
        bars = plt.bar(complexity_metrics, values, color=colors, alpha=0.7)
        plt.ylabel('Count / Percentage')
        plt.title('Model Complexity Metrics')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/svm_decision_boundary_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate accuracy on 2D projection
        accuracy_2d = svm_2d.score(X_eval_pca, self.y_eval)
        print(f"SVM accuracy on 2D PCA projection: {accuracy_2d:.3f}")
        print(f"Number of support vectors: {total_support_vectors} ({support_vector_ratio:.1%} of training data)")
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return {
            'pca_accuracy': accuracy_2d,
            'support_vectors': total_support_vectors,
            'support_vector_ratio': support_vector_ratio,
            'explained_variance': pca.explained_variance_ratio_.sum()
        }

    def feature_importance_analysis(self, best_model):
        """Analyze feature importance for linear SVM"""
        print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Train linear SVM for feature importance analysis
        linear_svm = SVC(C=best_model.C, kernel='linear', random_state=42)
        linear_svm.fit(self.X_train_scaled, self.y_train)
        
        # Get feature importance (coefficients)
        feature_importance = np.abs(linear_svm.coef_[0])
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        print("Top 15 most important features (Linear SVM):")
        print(feature_importance_df.tail(15))
        
        # Plot feature importance
        plt.figure(figsize=(12, 10))
        top_features = feature_importance_df.tail(15)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', alpha=0.7)
        plt.yticks(range(len(top_features)), [f.replace('_', ' ').title() for f in top_features['feature']])
        plt.xlabel('Feature Importance (|Coefficient|)')
        plt.title('Top 15 Feature Importance - Linear SVM')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(value + max(top_features['importance'])*0.01, i, f'{value:.3f}', 
                    va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/svm_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Compare linear vs RBF performance
        linear_score = cross_val_score(linear_svm, self.X_train_scaled, self.y_train, cv=5).mean()
        rbf_score = cross_val_score(best_model, self.X_train_scaled, self.y_train, cv=5).mean()
        
        print(f"\nLinear SVM CV accuracy: {linear_score:.3f}")
        print(f"RBF SVM CV accuracy: {rbf_score:.3f}")
        print(f"Performance difference: {rbf_score - linear_score:.3f}")
        
        return {
            'feature_importance_df': feature_importance_df,
            'linear_score': linear_score,
            'rbf_score': rbf_score
        }

    def final_svm_evaluation(self, best_model):
        """Evaluate final SVM model on test set if available"""
        if self.X_test is not None and self.y_test is not None:
            print(f"\n=== FINAL SVM MODEL EVALUATION ON TEST SET ===")
            
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
            plt.title('Confusion Matrix - Final SVM Model')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/confusion_matrix_final.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return test_accuracy
        
        return None

    def create_svm_performance_summary(self, c_results, gamma_results, kernel_results, grid_results):
        """Create comprehensive SVM performance summary"""
        print(f"\n=== SVM PERFORMANCE SUMMARY ===")
        
        # Create summary table
        summary_data = {
            'Method': ['C Optimization', 'Gamma Optimization', 'Kernel Comparison', 'Grid Search'],
            'Best_Parameter': [
                f"C={c_results['optimal_c']}", 
                f"Gamma={gamma_results['optimal_gamma']}", 
                f"{kernel_results['best_kernel']}",
                f"C={grid_results['best_params']['C']}, γ={grid_results['best_params']['gamma']}"
            ],
            'CV_Score': [
                c_results['best_cv_score'],
                gamma_results['best_gamma_score'],
                kernel_results['best_kernel_score'],
                grid_results['best_cv_score']
            ],
            f'{self.eval_set_name}_Score': [
                'N/A',  # Not calculated for individual parameter tests
                'N/A',
                'N/A',
                grid_results['eval_score']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print("\nSVM Performance Summary:")
        print(summary_df)
        
        summary_df.to_csv(f'{self.output_dir}/svm_performance_summary.csv', index=False)
        
        return summary_df

    def run_complete_svm_analysis(self):
        """Run complete SVM analysis"""
        print("="*70)
        print("           SUPPORT VECTOR MACHINE ANALYSIS")
        print("="*70)
        
        results = {}
        
        # 1. Test C parameter
        print("Step 1: Testing C parameter...")
        c_results = self.test_c_parameter()
        results['c_analysis'] = c_results
        
        # 2. Test gamma parameter
        print("\nStep 2: Testing gamma parameter...")
        gamma_results = self.test_gamma_parameter(c_results['optimal_c'])
        results['gamma_analysis'] = gamma_results
        
        # 3. Test different kernels
        print("\nStep 3: Testing different kernels...")
        kernel_results = self.test_kernel_comparison(c_results['optimal_c'])
        results['kernel_analysis'] = kernel_results
        
        # 4. Comprehensive grid search
        print("\nStep 4: Comprehensive grid search...")
        grid_results = self.comprehensive_grid_search_rbf()
        results['grid_search'] = grid_results
        
        # 5. Decision boundary analysis
        print("\nStep 5: Decision boundary analysis...")
        boundary_results = self.analyze_svm_decision_boundary(grid_results['best_model'])
        results['boundary_analysis'] = boundary_results
        
        # 6. Feature importance analysis
        print("\nStep 6: Feature importance analysis...")
        importance_results = self.feature_importance_analysis(grid_results['best_model'])
        results['importance_analysis'] = importance_results
        
        # 7. Performance summary
        print("\nStep 7: Creating performance summary...")
        summary = self.create_svm_performance_summary(c_results, gamma_results, kernel_results, grid_results)
        results['summary'] = summary
        
        # 8. Final evaluation
        print("\nStep 8: Final model evaluation...")
        test_accuracy = self.final_svm_evaluation(grid_results['best_model'])
        if test_accuracy:
            results['test_accuracy'] = test_accuracy
        
        print(f"\n{'='*70}")
        print("SUPPORT VECTOR MACHINE ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"All results saved in: {self.output_dir}/")
        
        return results

def main():
    """Main function"""
    # File paths
    train_file = 'data/IPC_classification_features_train.csv'
    val_file = 'data/IPC_classification_features_val.csv'
    test_file = 'data/IPC_classification_features_test.csv'
    
    print("Initializing Support Vector Machine Analyzer...")
    
    svm_analyzer = SVMAnalyzer(train_file, val_file, test_file)
    
    results = svm_analyzer.run_complete_svm_analysis()
    
    return results

if __name__ == "__main__":
    results = main()
