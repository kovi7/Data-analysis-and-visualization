import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import warnings
from pathlib import Path
import decision_tree as dt # Import Decision Tree Analyzer

warnings.filterwarnings('ignore')

"""
Task 2.2b: Random Forest 
"""


class RandomForestAnalyzer(dt.DecisionTreeAnalyzer):
    def __init__(self, train_file, val_file=None, test_file=None, output_dir='results/random_forest'):
        """Initialize Random Forest Analyzer inheriting from Decision Tree Analyzer"""
        # Initialize parent class but change output directory
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.output_dir = output_dir
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Use parent's data loading method
        self.load_and_prepare_data()
        
        print("Random Forest Analyzer initialized - inheriting from Decision Tree Analyzer")

    def test_n_estimators(self, n_estimators_range=[10, 25, 50, 100, 200, 500]):
        """Test different numbers of trees in Random Forest"""
        print(f"\n=== TESTING NUMBER OF TREES (N_ESTIMATORS) ===")
        
        results = {}
        train_scores = []
        eval_scores = []
        cv_scores = []
        
        for n_trees in n_estimators_range:
            print(f"Testing {n_trees} trees...")
            
            # Create Random Forest with n_trees
            rf = RandomForestClassifier(
                n_estimators=n_trees, 
                random_state=42, 
                n_jobs=-1
            )
            rf.fit(self.X_train, self.y_train)
            
            # Calculate scores
            train_score = rf.score(self.X_train, self.y_train)
            eval_score = rf.score(self.X_eval, self.y_eval)
            cv_score = cross_val_score(rf, self.X_train, self.y_train, cv=5, n_jobs=-1).mean()
            
            train_scores.append(train_score)
            eval_scores.append(eval_score)
            cv_scores.append(cv_score)
            
            results[n_trees] = {
                'train_score': train_score,
                'eval_score': eval_score,
                'cv_score': cv_score,
                'model': rf
            }
            
            print(f"Trees {n_trees:3d}: Train={train_score:.3f}, {self.eval_set_name}={eval_score:.3f}, CV={cv_score:.3f}")
        
        # Plot results using 4 subplots
        plt.figure(figsize=(16, 12))
        
        # Performance vs number of trees
        plt.subplot(2, 2, 1)
        plt.plot(n_estimators_range, train_scores, 'o-', label='Training', color='blue', linewidth=2)
        plt.plot(n_estimators_range, eval_scores, 'o-', label=f'{self.eval_set_name}', color='red', linewidth=2)
        plt.plot(n_estimators_range, cv_scores, 'o-', label='CV', color='green', linewidth=2)
        plt.xlabel('Number of Trees')
        plt.ylabel('Accuracy')
        plt.title('Random Forest Performance vs Number of Trees')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Overfitting analysis
        plt.subplot(2, 2, 2)
        overfitting = np.array(train_scores) - np.array(eval_scores)
        plt.plot(n_estimators_range, overfitting, 'o-', color='purple', linewidth=2)
        plt.xlabel('Number of Trees')
        plt.ylabel('Overfitting (Train - Eval)')
        plt.title('Overfitting Analysis')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Convergence analysis
        plt.subplot(2, 2, 3)
        score_diff = np.diff(cv_scores)
        plt.plot(n_estimators_range[1:], score_diff, 'o-', color='orange', linewidth=2)
        plt.xlabel('Number of Trees')
        plt.ylabel('CV Score Improvement')
        plt.title('Performance Convergence')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Best performance summary
        plt.subplot(2, 2, 4)
        best_cv_idx = np.argmax(cv_scores)
        best_eval_idx = np.argmax(eval_scores)
        
        categories = ['Best CV Trees', f'Best {self.eval_set_name} Trees', 'Best CV Score', f'Best {self.eval_set_name} Score']
        values = [n_estimators_range[best_cv_idx], n_estimators_range[best_eval_idx], 
                 cv_scores[best_cv_idx], eval_scores[best_eval_idx]]
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
        plt.savefig(f'{self.output_dir}/n_estimators_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        optimal_n_trees = n_estimators_range[best_cv_idx]
        print(f"\nOptimal number of trees (CV): {optimal_n_trees} (Score: {cv_scores[best_cv_idx]:.3f})")
        
        return {
            'results': results,
            'n_estimators_range': n_estimators_range,
            'train_scores': train_scores,
            'eval_scores': eval_scores,
            'cv_scores': cv_scores,
            'optimal_n_trees': optimal_n_trees,
            'best_cv_score': cv_scores[best_cv_idx]
        }

    def comprehensive_rf_grid_search(self):
        """Comprehensive grid search specifically for Random Forest"""
        print(f"\n=== COMPREHENSIVE RANDOM FOREST GRID SEARCH ===")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Performing Random Forest grid search... This may take several minutes.")
        grid_search.fit(self.X_train, self.y_train)
        
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        eval_score = best_model.score(self.X_eval, self.y_eval)
        
        print(f"\nBest Random Forest parameters: {best_params}")
        print(f"Best CV score: {best_cv_score:.3f}")
        print(f"{self.eval_set_name} accuracy: {eval_score:.3f}")
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'eval_score': eval_score
        }
    
    def compare_with_decision_tree(self, rf_model):
        """Compare Random Forest with Decision Tree using parent class methods"""
        print(f"\n=== RANDOM FOREST vs DECISION TREE COMPARISON ===")
        
        # Use parent class to create and train Decision Tree
        print("Training Decision Tree for comparison...")
        dt_results = super().test_tree_depths(range(5, 16))  # Test depths 5-15
        optimal_dt_depth = dt_results['optimal_depth']
        
        # Create optimal Decision Tree
        dt = DecisionTreeClassifier(max_depth=optimal_dt_depth, random_state=42)
        dt.fit(self.X_train, self.y_train)
        
        # Calculate scores for both models
        dt_train_score = dt.score(self.X_train, self.y_train)
        dt_eval_score = dt.score(self.X_eval, self.y_eval)
        dt_cv_score = cross_val_score(dt, self.X_train, self.y_train, cv=5).mean()
        
        rf_train_score = rf_model.score(self.X_train, self.y_train)
        rf_eval_score = rf_model.score(self.X_eval, self.y_eval)
        rf_cv_score = cross_val_score(rf_model, self.X_train, self.y_train, cv=5).mean()
        
        # Create comparison DataFrame
        comparison_data = {
            'Model': ['Decision Tree', 'Random Forest'],
            'Train_Score': [dt_train_score, rf_train_score],
            f'{self.eval_set_name}_Score': [dt_eval_score, rf_eval_score],
            'CV_Score': [dt_cv_score, rf_cv_score],
            'Overfitting': [dt_train_score - dt_eval_score, rf_train_score - rf_eval_score]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Comparison:")
        print(comparison_df)
        
        # Create figure with proper spacing
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Performance comparison
        ax1 = plt.subplot(2, 3, 1)
        x = np.arange(len(comparison_df))
        width = 0.25
        
        bars1 = ax1.bar(x - width, comparison_df['Train_Score'], width, label='Train', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x, comparison_df[f'{self.eval_set_name}_Score'], width, label=self.eval_set_name, alpha=0.8, color='lightcoral')
        bars3 = ax1.bar(x + width, comparison_df['CV_Score'], width, label='CV', alpha=0.8, color='lightgreen')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison_df['Model'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 2. Overfitting comparison
        ax2 = plt.subplot(2, 3, 2)
        colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in comparison_df['Overfitting']]
        bars = ax2.bar(comparison_df['Model'], comparison_df['Overfitting'], color=colors, alpha=0.7)
        ax2.set_ylabel('Overfitting (Train - Eval)')
        ax2.set_title('Overfitting Comparison')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, comparison_df['Overfitting']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comparison_df['Overfitting'])*0.05, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Feature importance comparison
        ax3 = plt.subplot(2, 3, 3)
        dt_importance = dt.feature_importances_
        rf_importance = rf_model.feature_importances_
        
        # Get top 10 features from RF
        top_indices = np.argsort(rf_importance)[-10:]
        top_features = [self.feature_names[i] for i in top_indices]
        
        y_pos = np.arange(len(top_features))
        ax3.barh(y_pos - 0.2, dt_importance[top_indices], 0.4, alpha=0.7, label='Decision Tree', color='lightblue')
        ax3.barh(y_pos + 0.2, rf_importance[top_indices], 0.4, alpha=0.7, label='Random Forest', color='lightgreen')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f.replace('_', ' ').title() for f in top_features])
        ax3.set_xlabel('Feature Importance')
        ax3.set_title('Top 10 Feature Importance Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Decision Tree Confusion Matrix
        ax4 = plt.subplot(2, 3, 4)
        dt_pred = dt.predict(self.X_eval)
        cm_dt = confusion_matrix(self.y_eval, dt_pred)
        sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-acidic', 'Acidic'],
                yticklabels=['Non-acidic', 'Acidic'],
                ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('Decision Tree Confusion Matrix')
        
        # 5. Random Forest Confusion Matrix
        ax5 = plt.subplot(2, 3, 5)
        rf_pred = rf_model.predict(self.X_eval)
        cm_rf = confusion_matrix(self.y_eval, rf_pred)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Non-acidic', 'Acidic'],
                yticklabels=['Non-acidic', 'Acidic'],
                ax=ax5, cbar_kws={'shrink': 0.8})
        ax5.set_title('Random Forest Confusion Matrix')
        
        # 6. Model complexity comparison
        ax6 = plt.subplot(2, 3, 6)
        
        
        # Calculate complexity metrics
        dt_nodes = dt.tree_.node_count
        rf_total_nodes = sum(tree.tree_.node_count for tree in rf_model.estimators_)
        
        complexity_data = {
            'Model': ['Decision Tree', 'Random Forest'],
            'Trees': [1, rf_model.n_estimators],
            'Total_Nodes': [dt_nodes, rf_total_nodes]
        }
        
        x_pos = np.arange(len(complexity_data['Model']))
        
        # Use two different y-axes for different scales
        ax6_twin = ax6.twinx()
        
        # Plot number of trees
        bars1 = ax6.bar(x_pos - 0.2, complexity_data['Trees'], 0.4, 
                    label='Number of Trees', alpha=0.7, color='orange')
        
        # Plot total nodes (in thousands) on secondary axis
        bars2 = ax6_twin.bar(x_pos + 0.2, np.array(complexity_data['Total_Nodes'])/1000, 0.4, 
                            label='Total Nodes (thousands)', alpha=0.7, color='purple')
        
        ax6.set_xlabel('Models')
        ax6.set_ylabel('Number of Trees', color='orange')
        ax6_twin.set_ylabel('Total Nodes (thousands)', color='purple')
        ax6.set_title('Model Complexity')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(complexity_data['Model'])
        
        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax6.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                    f'{complexity_data["Trees"][i]}', ha='center', va='bottom', fontsize=9)
            ax6_twin.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + bar2.get_height()*0.05,
                        f'{complexity_data["Total_Nodes"][i]/1000:.1f}k', ha='center', va='bottom', fontsize=9)
        
        # Add legends
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Adjust layout to prevent overlap
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Save with high quality
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.show()
        
        return comparison_df
    
    def visualize_random_forest_trees(self, rf_model, n_trees=3, max_depth=3):
        """Visualize individual trees from Random Forest"""
        print(f"\n=== VISUALIZING RANDOM FOREST TREES ===")
        
        print(f"Visualizing first {n_trees} trees using plot_tree...")
        
        fig, axes = plt.subplots(nrows=1, ncols=n_trees, figsize=(20, 8))
        if n_trees == 1:
            axes = [axes]
        
        for i in range(n_trees):
            tree = rf_model.estimators_[i]
            
            plot_tree(tree,
                    ax=axes[i],
                    feature_names=[f.replace('_', ' ').title() for f in self.feature_names],
                    class_names=['Non-acidic (pI >= 5.0)', 'Acidic (pI < 5.0)'],
                    filled=True,
                    rounded=True,
                    max_depth=max_depth,
                    fontsize=8)
            
            axes[i].set_title(f'Tree {i+1} from Random Forest', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/random_forest_trees.png', dpi=300, bbox_inches='tight')
        
    def analyze_tree_diversity(self, rf_model, n_trees=5):
        """Analyze diversity between trees in Random Forest"""
        print(f"\n=== ANALYZING TREE DIVERSITY ===")
        
        # Compare feature importance across different trees
        tree_importances = []
        tree_depths = []
        tree_nodes = []
        
        for i, tree in enumerate(rf_model.estimators_[:n_trees]):
            importance = tree.feature_importances_
            tree_importances.append(importance)
            tree_depths.append(tree.tree_.max_depth)
            tree_nodes.append(tree.tree_.node_count)
        
        tree_importances = np.array(tree_importances)
        
        # Plot tree characteristics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Tree depths
        axes[0, 0].bar(range(1, n_trees+1), tree_depths, color='skyblue', alpha=0.7)
        axes[0, 0].set_xlabel('Tree Number')
        axes[0, 0].set_ylabel('Max Depth')
        axes[0, 0].set_title('Tree Depths in Random Forest')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Tree node counts
        axes[0, 1].bar(range(1, n_trees+1), tree_nodes, color='lightcoral', alpha=0.7)
        axes[0, 1].set_xlabel('Tree Number')
        axes[0, 1].set_ylabel('Number of Nodes')
        axes[0, 1].set_title('Tree Complexity (Node Count)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature importance correlation between trees
        importance_corr = np.corrcoef(tree_importances)
        im = axes[1, 0].imshow(importance_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Feature Importance Correlation Between Trees')
        axes[1, 0].set_xlabel('Tree Number')
        axes[1, 0].set_ylabel('Tree Number')
        
        # Add correlation values
        for i in range(n_trees):
            for j in range(n_trees):
                axes[1, 0].text(j, i, f'{importance_corr[i, j]:.2f}', 
                            ha='center', va='center', 
                            color='white' if abs(importance_corr[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        
        # Top features used by each tree
        axes[1, 1].boxplot([tree_importances[:, i] for i in range(min(10, len(self.feature_names)))],
                        labels=[f.replace('_', ' ')[:8] for f in self.feature_names[:10]])
        axes[1, 1].set_title('Feature Importance Distribution Across Trees')
        axes[1, 1].set_ylabel('Importance')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/tree_diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print diversity statistics
        print(f"Tree Diversity Statistics:")
        print(f"Average tree depth: {np.mean(tree_depths):.2f} ± {np.std(tree_depths):.2f}")
        print(f"Average nodes per tree: {np.mean(tree_nodes):.0f} ± {np.std(tree_nodes):.0f}")
        print(f"Average feature importance correlation: {np.mean(importance_corr[np.triu_indices_from(importance_corr, k=1)]):.3f}")
        
        return {
            'tree_depths': tree_depths,
            'tree_nodes': tree_nodes,
            'importance_correlation': importance_corr,
            'tree_importances': tree_importances
        }

    def compare_best_vs_worst_trees(self, rf_model):
        """Compare the best and worst performing trees in the forest"""
        print(f"\n=== COMPARING BEST vs WORST TREES ===")
        
        # Calculate individual tree performance
        tree_scores = []
        for tree in rf_model.estimators_:
            score = tree.score(self.X_eval, self.y_eval)
            tree_scores.append(score)
        
        tree_scores = np.array(tree_scores)
        best_tree_idx = np.argmax(tree_scores)
        worst_tree_idx = np.argmin(tree_scores)
        
        print(f"Best tree (#{best_tree_idx}): {tree_scores[best_tree_idx]:.3f} accuracy")
        print(f"Worst tree (#{worst_tree_idx}): {tree_scores[worst_tree_idx]:.3f} accuracy")
        print(f"Random Forest ensemble: {rf_model.score(self.X_eval, self.y_eval):.3f} accuracy")
        
        # Visualize best and worst trees
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Best tree
        plot_tree(rf_model.estimators_[best_tree_idx],
                ax=axes[0],
                feature_names=[f.replace('_', ' ').title() for f in self.feature_names],
                class_names=['Non-acidic', 'Acidic'],
                filled=True,
                rounded=True,
                max_depth=4,
                fontsize=8)
        axes[0].set_title(f'Best Tree (#{best_tree_idx}) - Accuracy: {tree_scores[best_tree_idx]:.3f}', 
                        fontsize=14, fontweight='bold')
        
        # Worst tree
        plot_tree(rf_model.estimators_[worst_tree_idx],
                ax=axes[1],
                feature_names=[f.replace('_', ' ').title() for f in self.feature_names],
                class_names=['Non-acidic', 'Acidic'],
                filled=True,
                rounded=True,
                max_depth=4,
                fontsize=8)
        axes[1].set_title(f'Worst Tree (#{worst_tree_idx}) - Accuracy: {tree_scores[worst_tree_idx]:.3f}', 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/best_vs_worst_trees.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot distribution of tree scores
        plt.figure(figsize=(10, 6))
        plt.hist(tree_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(tree_scores[best_tree_idx], color='green', linestyle='--', linewidth=2, label=f'Best: {tree_scores[best_tree_idx]:.3f}')
        plt.axvline(tree_scores[worst_tree_idx], color='red', linestyle='--', linewidth=2, label=f'Worst: {tree_scores[worst_tree_idx]:.3f}')
        plt.axvline(rf_model.score(self.X_eval, self.y_eval), color='orange', linestyle='-', linewidth=3, label=f'Ensemble: {rf_model.score(self.X_eval, self.y_eval):.3f}')
        
        plt.xlabel('Individual Tree Accuracy')
        plt.ylabel('Number of Trees')
        plt.title('Distribution of Individual Tree Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/tree_performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'tree_scores': tree_scores,
            'best_tree_idx': best_tree_idx,
            'worst_tree_idx': worst_tree_idx,
            'ensemble_score': rf_model.score(self.X_eval, self.y_eval)
        }

    def analyze_rf_feature_importance(self, model):
        """Analyze Random Forest feature importance (extends parent method)"""
        print(f"\n=== RANDOM FOREST FEATURE IMPORTANCE ANALYSIS ===")
        
        # Use parent class method but customize for Random Forest
        feature_importance_df = super().analyze_feature_importance(model, "Random Forest")
        
        # Additional Random Forest specific analysis
        print(f"\nRandom Forest specific insights:")
        print(f"Number of trees: {model.n_estimators}")
        print(f"Out-of-bag score: {model.oob_score_ if hasattr(model, 'oob_score_') else 'Not available'}")
        
        # Analyze feature importance stability across trees
        if hasattr(model, 'estimators_'):
            tree_importances = np.array([tree.feature_importances_ for tree in model.estimators_])
            importance_std = np.std(tree_importances, axis=0)
            
            # Plot importance stability
            plt.figure(figsize=(12, 8))
            top_features_idx = np.argsort(feature_importance_df['importance'].values)[-15:]
            
            plt.errorbar(range(len(top_features_idx)), 
                        feature_importance_df['importance'].values[top_features_idx],
                        yerr=importance_std[top_features_idx],
                        fmt='o', capsize=5, capthick=2)
            
            plt.xticks(range(len(top_features_idx)), 
                      [self.feature_names[i].replace('_', ' ').title() for i in top_features_idx], 
                      rotation=45, ha='right')
            plt.ylabel('Feature Importance')
            plt.title('Feature Importance with Stability (Error Bars)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/feature_importance_stability.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return feature_importance_df

    def create_rf_performance_summary(self, n_trees_results, grid_results, comparison_df):
        """Create comprehensive Random Forest performance summary"""
        print(f"\n=== RANDOM FOREST PERFORMANCE SUMMARY ===")
        
        # Create detailed summary table
        summary_data = {
            'Method': ['N-Trees Optimization', 'Grid Search', 'vs Decision Tree'],
            'Best_Parameter': [
                f"Trees: {n_trees_results['optimal_n_trees']}", 
                f"Multiple: {grid_results['best_params']}",
                f"Improvement: {comparison_df.iloc[1]['CV_Score'] - comparison_df.iloc[0]['CV_Score']:.3f}"
            ],
            'CV_Score': [
                n_trees_results['best_cv_score'],
                grid_results['best_cv_score'],
                comparison_df.iloc[1]['CV_Score']
            ],
            f'{self.eval_set_name}_Score': [
                n_trees_results['results'][n_trees_results['optimal_n_trees']]['eval_score'],
                grid_results['eval_score'],
                comparison_df.iloc[1][f'{self.eval_set_name}_Score']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print("\nRandom Forest Performance Summary:")
        print(summary_df)
        
        # Save detailed results
        summary_df.to_csv(f'{self.output_dir}/rf_performance_summary.csv', index=False)
        comparison_df.to_csv(f'{self.output_dir}/rf_vs_dt_comparison.csv', index=False)
        
        return summary_df

    def run_complete_rf_analysis(self):
        """Run complete Random Forest analysis inheriting and extending Decision Tree methods"""
        print("="*70)
        print("           RANDOM FOREST ANALYSIS")
        print("           (Inheriting from Decision Tree)")
        print("="*70)
        
        results = {}
        
        # 1. Test number of trees (Random Forest specific)
        print("Step 1: Testing number of trees...")
        n_trees_results = self.test_n_estimators()
        results['n_trees_analysis'] = n_trees_results
        
        # 2. Random Forest specific grid search
        print("\nStep 2: Random Forest grid search...")
        grid_results = self.comprehensive_rf_grid_search()
        results['grid_search'] = grid_results
        
        # 3. Feature importance analysis (using parent method)
        print("\nStep 3: Feature importance analysis...")
        feature_importance = self.analyze_rf_feature_importance(grid_results['best_model'])
        results['feature_importance'] = feature_importance
        
        # 4. Compare with Decision Tree (using parent methods)
        print("\nStep 4: Comparing with Decision Tree...")
        comparison = self.compare_with_decision_tree(grid_results['best_model'])
        results['comparison'] = comparison
        
        # 5. Performance summary
        print("\nStep 5: Creating performance summary...")
        summary = self.create_rf_performance_summary(n_trees_results, grid_results, comparison)
        results['summary'] = summary
        
        # 6. Final evaluation (using parent method)
        print("\nStep 6: Final model evaluation...")
        test_accuracy = super().final_model_evaluation(grid_results['best_model'])
        if test_accuracy:
            results['test_accuracy'] = test_accuracy

         # 7. Visualize Random Forest trees (NEW)
        print("\nStep 7: Visualizing Random Forest trees...")
        self.visualize_random_forest_trees(grid_results['best_model'], n_trees=3, max_depth=3)
        
        # 8. Analyze tree diversity (NEW)
        print("\nStep 8: Analyzing tree diversity...")
        diversity_results = self.analyze_tree_diversity(grid_results['best_model'], n_trees=5)
        results['diversity_analysis'] = diversity_results
        
        # 9. Compare best vs worst trees (NEW)
        print("\nStep 9: Comparing best vs worst trees...")
        tree_comparison = self.compare_best_vs_worst_trees(grid_results['best_model'])
        results['tree_comparison'] = tree_comparison
        
        print(f"\n{'='*70}")
        print("RANDOM FOREST ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"All results saved in: {self.output_dir}/")
        print(f"Inherited methods from Decision Tree Analyzer successfully used!")
        
        return results

def main():
    """Main function"""
    # File paths
    train_file = 'data/IPC_classification_features_train.csv'
    val_file = 'data/IPC_classification_features_val.csv'
    test_file = 'data/IPC_classification_features_test.csv'
    
    # Initialize Random Forest analyzer (inheriting from Decision Tree)
    print("Initializing Random Forest Analyzer...")
    print("This analyzer inherits all methods from Decision Tree Analyzer")
    
    rf_analyzer = RandomForestAnalyzer(train_file, val_file, test_file)
    
    # Run complete analysis
    results = rf_analyzer.run_complete_rf_analysis()
    
    # Demonstrate inheritance by calling parent methods directly
    print(f"\n{'='*50}")
    print("DEMONSTRATING INHERITANCE:")
    print(f"{'='*50}")
    print("Calling parent class methods directly...")
    
    # Example: Use parent's tree depth testing method
    print("\nUsing inherited tree depth testing method:")
    depth_results = rf_analyzer.test_tree_depths(range(3, 8))
    print(f"Optimal depth from parent method: {depth_results['optimal_depth']}")
    
    return results

if __name__ == "__main__":
    results = main()
