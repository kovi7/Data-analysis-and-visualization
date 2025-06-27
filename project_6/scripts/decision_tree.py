import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV,  cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

"""
Task 2.2a: Decision Tree Classifier Implementation
Testing different tree structures and visualizing results
"""

class DecisionTreeAnalyzer:
    def __init__(self, train_file, val_file=None, test_file=None, output_dir='results/decision_tree'):
        """Initialize with CSV file paths and output directory"""
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(exist_ok=True)
        
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and prepare data for analysis"""
        print("=== DECISION TREE DATA PREPARATION ===")
        
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
            self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
            )
            self.eval_set_name = "Holdout"
        
        print(f"Using {self.eval_set_name} set for evaluation")

    def test_tree_depths(self, max_depth_range=range(1, 21)):
        """Test different tree depths"""
        print(f"\n=== TESTING TREE DEPTH PARAMETER ===")
        
        train_scores = []
        eval_scores = []
        cv_scores = []
        
        for depth in max_depth_range:
            # Create and train classifier
            clf = DecisionTreeClassifier(
                max_depth=depth,
                random_state=42,
                criterion='entropy'
            )
            clf.fit(self.X_train, self.y_train)
            
            # Calculate scores
            train_score = clf.score(self.X_train, self.y_train)
            eval_score = clf.score(self.X_eval, self.y_eval)
            cv_score = cross_val_score(clf, self.X_train, self.y_train, cv=5).mean()
            
            train_scores.append(train_score)
            eval_scores.append(eval_score)
            cv_scores.append(cv_score)
            
            print(f"Depth {depth:2d}: Train={train_score:.3f}, {self.eval_set_name}={eval_score:.3f}, CV={cv_score:.3f}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Main performance plot
        plt.subplot(2, 2, 1)
        plt.plot(max_depth_range, train_scores, 'o-', label='Training Accuracy', color='blue')
        plt.plot(max_depth_range, eval_scores, 'o-', label=f'{self.eval_set_name} Accuracy', color='red')
        plt.plot(max_depth_range, cv_scores, 'o-', label='CV Accuracy', color='green')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title(f'Decision Tree Performance vs Max Depth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(max_depth_range)
        
        # Overfitting analysis
        plt.subplot(2, 2, 2)
        overfitting = np.array(train_scores) - np.array(eval_scores)
        plt.plot(max_depth_range, overfitting, 'o-', color='purple')
        plt.xlabel('Max Depth')
        plt.ylabel('Overfitting (Train - Eval)')
        plt.title('Overfitting Analysis')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(max_depth_range)
        
        # Best depths
        plt.subplot(2, 2, 3)
        best_eval_idx = np.argmax(eval_scores)
        best_cv_idx = np.argmax(cv_scores)
        
        plt.bar(['Best Eval', 'Best CV'], 
                [max_depth_range[best_eval_idx], max_depth_range[best_cv_idx]], 
                color=['red', 'green'], alpha=0.7)
        plt.ylabel('Optimal Depth')
        plt.title('Optimal Depths')
        
        # Performance summary
        plt.subplot(2, 2, 4)
        best_scores = [eval_scores[best_eval_idx], cv_scores[best_cv_idx]]
        plt.bar([f'Best {self.eval_set_name}', 'Best CV'], best_scores, 
                color=['red', 'green'], alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title('Best Scores')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/depth_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        optimal_depth = max_depth_range[best_cv_idx]
        print(f"\nOptimal depth (CV): {optimal_depth} (Score: {cv_scores[best_cv_idx]:.3f})")
        
        return {
            'depths': list(max_depth_range),
            'train_scores': train_scores,
            'eval_scores': eval_scores,
            'cv_scores': cv_scores,
            'optimal_depth': optimal_depth,
            'best_cv_score': cv_scores[best_cv_idx]
        }

    def test_number_of_features(self, optimal_depth=10):
        """Test different numbers of features using feature selection"""
        print(f"\n=== TESTING NUMBER OF FEATURES ===")
        
        max_features = min(20, len(self.feature_names))
        feature_range = range(1, max_features + 1)
        feature_scores = []
        selected_features_list = []
        
        for n_features in feature_range:
            # Select top k features
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_train_selected = selector.fit_transform(self.X_train, self.y_train)
            X_eval_selected = selector.transform(self.X_eval)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            selected_features_list.append(selected_features)
            
            # Train classifier with selected features
            clf = DecisionTreeClassifier(
                max_depth=optimal_depth,
                random_state=42,
                criterion='entropy'
            )
            clf.fit(X_train_selected, self.y_train)
            
            # Calculate score
            eval_score = clf.score(X_eval_selected, self.y_eval)
            feature_scores.append(eval_score)
            
            print(f"Features {n_features:2d}: {self.eval_set_name} Accuracy = {eval_score:.3f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(feature_range, feature_scores, 'o-', color='green')
        plt.xlabel('Number of Features')
        plt.ylabel(f'{self.eval_set_name} Accuracy')
        plt.title('Decision Tree Performance vs Number of Features')
        plt.grid(True, alpha=0.3)
        plt.xticks(feature_range)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/features_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        optimal_n_features = feature_range[np.argmax(feature_scores)]
        optimal_features = selected_features_list[optimal_n_features - 1]
        
        print(f"\nOptimal number of features: {optimal_n_features} (Score: {max(feature_scores):.3f})")
        print(f"Selected features: {optimal_features}")
        
        return {
            'feature_range': list(feature_range),
            'feature_scores': feature_scores,
            'optimal_n_features': optimal_n_features,
            'optimal_features': optimal_features
        }

    def comprehensive_grid_search(self):
        """Perform comprehensive grid search for hyperparameter tuning"""
        print(f"\n=== COMPREHENSIVE GRID SEARCH ===")
        
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        }
        
        clf = DecisionTreeClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            clf, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Performing grid search...")
        grid_search.fit(self.X_train, self.y_train)
        
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        eval_score = best_model.score(self.X_eval, self.y_eval)
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best CV score: {best_cv_score:.3f}")
        print(f"{self.eval_set_name} accuracy: {eval_score:.3f}")
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'eval_score': eval_score
        }

    def visualize_decision_trees(self, optimal_depth=5, optimal_features=None):
        """Create visualizations of decision trees"""
        print(f"\n=== VISUALIZING DECISION TREES ===")
        
        # Simple tree visualization
        clf_simple = DecisionTreeClassifier(max_depth=3, random_state=42, criterion='entropy')
        clf_simple.fit(self.X_train, self.y_train)
        
        plt.figure(figsize=(10, 8))
        plot_tree(clf_simple, 
                  feature_names=[f.replace('_', ' ').title() for f in self.feature_names],
                  class_names=['Non-acidic (pI >= 5.0)', 'Acidic (pI < 5.0)'],
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title('Decision Tree - Simple (Max Depth = 3)')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/decision_tree_simple.png', bbox_inches='tight')
        plt.show()
        
        # Optimal depth tree
        clf_optimal = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42, criterion='entropy')
        clf_optimal.fit(self.X_train, self.y_train)
        
        plt.figure(figsize=(10, 8))
        plot_tree(clf_optimal,
                  feature_names=[f.replace('_', ' ').title() for f in self.feature_names],
                  class_names=['Non-acidic (pI >= 5.0)', 'Acidic (pI < 5.0)'],
                  filled=True,
                  rounded=True,
                  fontsize=8)
        plt.title(f'Decision Tree - Optimal Depth ({optimal_depth})')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/decision_tree_optimal.png', bbox_inches='tight')
        plt.show()
        
        # Tree with selected features
        if optimal_features and len(optimal_features) < len(self.feature_names):
            feature_indices = [i for i, name in enumerate(self.feature_names) if name in optimal_features]
            X_train_selected = self.X_train.iloc[:, feature_indices]
            
            clf_selected = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42, criterion='entropy')
            clf_selected.fit(X_train_selected, self.y_train)
            
            plt.figure(figsize=(10, 8))
            plot_tree(clf_selected,
                      feature_names=[f.replace('_', ' ').title() for f in optimal_features],
                      class_names=['Non-acidic (pI >= 5.0)', 'Acidic (pI < 5.0)'],
                      filled=True,
                      rounded=True,
                      fontsize=10)
            plt.title(f'Decision Tree - Selected Features ({len(optimal_features)} features)')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/decision_tree_selected_features.png', bbox_inches='tight')
            plt.show()

    def analyze_feature_importance(self, model=None, model_name="Decision Tree"):
        """Analyze and visualize feature importance"""
        print(f"\n=== FEATURE IMPORTANCE ANALYSIS - {model_name} ===")
        
        # Jeśli nie podano modelu, stwórz nowy Decision Tree
        if model is None:
            model = DecisionTreeClassifier(max_depth=10, random_state=42, criterion='entropy')
            model.fit(self.X_train, self.y_train)
        
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"Top 15 most important features for {model_name}:")
        print(feature_importance_df.head(15))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), [f.replace('_', ' ').title() for f in top_features['feature']])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df

    def create_performance_summary(self, depth_results, feature_results, grid_results):
        """Create comprehensive performance summary"""
        print(f"\n=== PERFORMANCE SUMMARY ===")
        
        # Create summary table
        summary_data = {
            'Method': ['Depth Optimization', 'Feature Selection', 'Grid Search'],
            'Best_Parameter': [
                f"Depth: {depth_results['optimal_depth']}", 
                f"Features: {feature_results['optimal_n_features']}", 
                f"Multiple: {grid_results['best_params']}"
            ],
            'CV_Score': [
                depth_results['best_cv_score'],
                max(feature_results['feature_scores']),
                grid_results['best_cv_score']
            ],
            f'{self.eval_set_name}_Score': [
                depth_results['eval_scores'][depth_results['optimal_depth']-1],
                max(feature_results['feature_scores']),
                grid_results['eval_score']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print("\nPerformance Summary:")
        print(summary_df)
        
        # Save to CSV
        summary_df.to_csv(f'{self.output_dir}/performance_summary.csv', index=False)
        
        return summary_df

    def final_model_evaluation(self, best_model):
        """Evaluate final model on test set if available"""
        if self.X_test is not None and self.y_test is not None:
            print(f"\n=== FINAL MODEL EVALUATION ON TEST SET ===")
            
            y_pred = best_model.predict(self.X_test)
            test_accuracy = accuracy_score(self.y_test, y_pred)
            
            print(f"Final test accuracy: {test_accuracy:.3f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Non-acidic', 'Acidic']))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Non-acidic', 'Acidic'],
                        yticklabels=['Non-acidic', 'Acidic'])
            plt.title('Confusion Matrix - Final Decision Tree Model')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/confusion_matrix_final.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return test_accuracy
        
        return None

    def run_complete_analysis(self):
        """Run complete decision tree analysis"""
        print("="*60)
        print("           DECISION TREE ANALYSIS")
        print("="*60)
        
        results = {}
        
        # 1. Test tree depths
        depth_results = self.test_tree_depths()
        results['depth_analysis'] = depth_results
        
        # 2. Test number of features
        feature_results = self.test_number_of_features(depth_results['optimal_depth'])
        results['feature_analysis'] = feature_results
        
        # 3. Grid search
        grid_results = self.comprehensive_grid_search()
        results['grid_search'] = grid_results
        
        # 4. Visualize trees
        self.visualize_decision_trees(
            depth_results['optimal_depth'], 
            feature_results['optimal_features']
        )
        
        # 5. Feature importance - przekaż model z grid search
        feature_importance = self.analyze_feature_importance(grid_results['best_model'], "Decision Tree")
        results['feature_importance'] = feature_importance
        
        # 6. Performance summary
        summary = self.create_performance_summary(depth_results, feature_results, grid_results)
        results['summary'] = summary
        
        # 7. Final evaluation
        test_accuracy = self.final_model_evaluation(grid_results['best_model'])
        if test_accuracy:
            results['test_accuracy'] = test_accuracy
        
        print(f"\n{'='*60}")
        print("DECISION TREE ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"All results saved in: {self.output_dir}/")
        
        return results


def main():
    """Main function"""
    # File paths
    train_file = 'data/IPC_classification_features_train.csv'
    val_file = 'data/IPC_classification_features_val.csv'
    test_file = 'data/IPC_classification_features_test.csv'
    
    # Initialize analyzer
    analyzer = DecisionTreeAnalyzer(train_file, val_file, test_file)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()
