import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

class DataExplorer:
    def __init__(self, csv_file, output_dir='results/features'):
        """Initialize with CSV file path and output directory for plots"""
        self.df = pd.read_csv(csv_file)
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for analysis"""
        print("Dataset info:")
        
        # Identify feature types
        self.categorical_features = ['data_type']
        self.target = 'label'
        
        # All numeric features (excluding uid and target)
        self.numeric_features = [col for col in self.df.columns 
                               if col not in ['uid', 'data_type', 'label']]
        
        # Amino acid features
        self.aa_features = [col for col in self.df.columns if col.startswith('aa_')]
        
        # Charge features
        self.charge_features = [col for col in self.df.columns if col.startswith('charge_')]
        
        # Other features
        self.other_features = [col for col in self.numeric_features 
                             if col not in self.aa_features + self.charge_features]
        
        print(f"\nFeature categories:")
        print(f"Amino acid features: {len(self.aa_features)}")
        print(f"Charge features: {len(self.charge_features)}")
        print(f"Other features: {len(self.other_features)}")
        
    def plot_target_distribution(self):
        """Plot target variable distribution"""
        plt.figure(figsize=(15, 5))
        
        # Overall distribution
        plt.subplot(1, 3, 1)
        self.df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Target Distribution\n(0: Non-acidic, 1: Acidic)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Distribution by data type
        plt.subplot(1, 3, 2)
        pd.crosstab(self.df['data_type'], self.df['label']).plot(kind='bar', 
                                                                color=['skyblue', 'salmon'])
        plt.title('Target Distribution by Data Type')
        plt.ylabel('Count')
        plt.legend(['Non-acidic', 'Acidic'])
        
        # Percentage by data type
        plt.subplot(1, 3, 3)
        ct = pd.crosstab(self.df['data_type'], self.df['label'], normalize='index') * 100
        ct.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Target Percentage by Data Type')
        plt.ylabel('Percentage')
        plt.legend(['Non-acidic', 'Acidic'])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/01_target_distribution.png")
        
    def plot_numeric_features(self, features, title_prefix="", file_prefix=""):
        """Create multiple plots for numeric features"""
        n_features = len(features)
        if n_features == 0:
            return
            
        # Calculate subplot layout
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Histograms
        plt.figure(figsize=(4*n_cols, 4*n_rows))
        for i, feature in enumerate(features):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Plot histograms for each class
            for label in [0, 1]:
                data = self.df[self.df['label'] == label][feature]
                plt.hist(data, alpha=0.7, label=f'Label {label}', bins=20)
            
            plt.title(f'{title_prefix}{feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{file_prefix}_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/{file_prefix}_histograms.png")
        
        # Box plots
        plt.figure(figsize=(4*n_cols, 4*n_rows))
        for i, feature in enumerate(features):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Create box plot
            data_to_plot = [self.df[self.df['label'] == label][feature] for label in [0, 1]]
            box_plot = plt.boxplot(data_to_plot, labels=['Non-acidic', 'Acidic'], patch_artist=True)
            
            # Color the boxes
            colors = ['skyblue', 'salmon']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            plt.title(f'{title_prefix}{feature}')
            plt.ylabel(feature)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{file_prefix}_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/{file_prefix}_boxplots.png")
        
    
    def plot_correlation_analysis(self):
        """Comprehensive correlation analysis"""
        # Correlation with target
        correlations = self.df[self.numeric_features + [self.target]].corr()[self.target].drop(self.target)
        correlations_abs = correlations.abs().sort_values(ascending=False)
        
        # Plot correlation with target
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        top_correlations = correlations_abs.head(15)
        colors = ['red' if correlations[feature] < 0 else 'blue' for feature in top_correlations.index]
        plt.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7)
        plt.yticks(range(len(top_correlations)), top_correlations.index)
        plt.xlabel('Absolute Correlation with Target')
        plt.title('Top 15 Features by Correlation with Target')
        plt.grid(True, alpha=0.3)
        
        # Correlation heatmap for top features
        plt.subplot(2, 2, 2)
        top_features = correlations_abs.head(10).index.tolist() + [self.target]
        corr_matrix = self.df[top_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Heatmap - Top 10 Features')
        
        # Feature correlation scatter plot (top 2 vs target)
        if len(correlations_abs) >= 2:
            plt.subplot(2, 2, 3)
            feature1 = correlations_abs.index[0]
            colors = ['skyblue' if label == 0 else 'salmon' for label in self.df['label']]
            plt.scatter(self.df[feature1], self.df['label'], c=colors, alpha=0.6)
            plt.xlabel(feature1)
            plt.ylabel('Label')
            plt.title(f'Scatter: {feature1} vs Target')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            feature2 = correlations_abs.index[1]
            plt.scatter(self.df[feature2], self.df['label'], c=colors, alpha=0.6)
            plt.xlabel(feature2)
            plt.ylabel('Label')
            plt.title(f'Scatter: {feature2} vs Target')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/05_correlation_analysis.png")
        
        return correlations_abs
    
    def plot_feature_relationships(self):
        """Plot relationships between different feature groups"""
        correlations_abs = self.df[self.numeric_features + [self.target]].corr()[self.target].drop(self.target).abs().sort_values(ascending=False)
        
        # Select top features from each category
        top_aa = [f for f in correlations_abs.head(10).index if f in self.aa_features][:3]
        top_charge = [f for f in correlations_abs.head(10).index if f in self.charge_features][:3]
        top_other = [f for f in correlations_abs.head(10).index if f in self.other_features][:3]
        
        # Pairplot of top features
        if len(top_aa) > 0 and len(top_charge) > 0:
            selected_features = (top_aa[:2] + top_charge[:2] + top_other[:2] + ['label'])[:7]
            
            plt.figure(figsize=(15, 12))
            
            # Create pairplot manually
            n_features = len(selected_features) - 1  # Exclude label
            fig, axes = plt.subplots(n_features, n_features, figsize=(15, 12))
            
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        # Diagonal: histograms
                        for label in [0, 1]:
                            data = self.df[self.df['label'] == label][selected_features[i]]
                            axes[i, j].hist(data, alpha=0.7, label=f'Label {label}', bins=15)
                        axes[i, j].set_title(selected_features[i])
                        if i == 0:
                            axes[i, j].legend()
                    else:
                        # Off-diagonal: scatter plots
                        for label in [0, 1]:
                            data = self.df[self.df['label'] == label]
                            color = 'skyblue' if label == 0 else 'salmon'
                            axes[i, j].scatter(data[selected_features[j]], 
                                             data[selected_features[i]], 
                                             c=color, alpha=0.5, s=10)
                        
                        if i == n_features - 1:
                            axes[i, j].set_xlabel(selected_features[j])
                        if j == 0:
                            axes[i, j].set_ylabel(selected_features[i])
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/06_feature_relationships.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {self.output_dir}/06_feature_relationships.png")
    
    def analyze_amino_acid_patterns(self):
        """Analyze amino acid composition patterns"""
        # Calculate amino acid percentages
        aa_data = self.df[self.aa_features].copy()
        aa_percentages = aa_data.div(self.df['length'], axis=0) * 100
        
        # Mean amino acid composition by class
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        mean_aa_0 = aa_percentages[self.df['label'] == 0].mean()
        mean_aa_1 = aa_percentages[self.df['label'] == 1].mean()
        
        x = np.arange(len(self.aa_features))
        width = 0.35
        
        aa_names = [f.replace('aa_', '') for f in self.aa_features]
        plt.bar(x - width/2, mean_aa_0, width, label='Non-acidic', color='skyblue', alpha=0.7)
        plt.bar(x + width/2, mean_aa_1, width, label='Acidic', color='salmon', alpha=0.7)
        
        plt.xlabel('Amino Acids')
        plt.ylabel('Mean Percentage')
        plt.title('Mean Amino Acid Composition by Class')
        plt.xticks(x, aa_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Difference in amino acid composition
        plt.subplot(2, 2, 2)
        aa_diff = mean_aa_1 - mean_aa_0
        colors = ['red' if diff < 0 else 'blue' for diff in aa_diff]
        plt.bar(x, aa_diff, color=colors, alpha=0.7)
        plt.xlabel('Amino Acids')
        plt.ylabel('Difference (Acidic - Non-acidic)')
        plt.title('Amino Acid Composition Difference')
        plt.xticks(x, aa_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Top discriminative amino acids
        plt.subplot(2, 2, 3)
        aa_correlations = self.df[self.aa_features + ['label']].corr()['label'].drop('label').abs().sort_values(ascending=False)
        top_aa_corr = aa_correlations.head(10)
        
        plt.barh(range(len(top_aa_corr)), top_aa_corr.values, color='green', alpha=0.7)
        plt.yticks(range(len(top_aa_corr)), [f.replace('aa_', '') for f in top_aa_corr.index])
        plt.xlabel('Absolute Correlation with Target')
        plt.title('Top 10 Discriminative Amino Acids')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_amino_acid_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/03_amino_acid_analysis.png")
        
        return aa_correlations
    
    def analyze_charge_features(self):
        """Analyze charge-related features"""
        plt.figure(figsize=(15, 10))
        
        # Charge features correlation
        plt.subplot(2, 2, 1)
        charge_corr = self.df[self.charge_features].corr()
        sns.heatmap(charge_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Between Charge Scales')
        
        # Charge vs pI relationship
        plt.subplot(2, 2, 2)
        # Select one representative charge feature
        charge_feature = self.charge_features[0]  # Sillero scale
        colors = ['skyblue' if label == 0 else 'salmon' for label in self.df['label']]
        plt.scatter(self.df['pI'], self.df[charge_feature], c=colors, alpha=0.6)
        plt.xlabel('pI')
        plt.ylabel(charge_feature)
        plt.title('Charge vs pI Relationship')
        plt.grid(True, alpha=0.3)
        
        # Charge feature importance
        plt.subplot(2, 2, 3)
        charge_correlations = self.df[self.charge_features + ['label']].corr()['label'].drop('label').abs().sort_values(ascending=False)
        
        plt.barh(range(len(charge_correlations)), charge_correlations.values, color='purple', alpha=0.7)
        plt.yticks(range(len(charge_correlations)), 
                   [f.replace('charge_', '') for f in charge_correlations.index])
        plt.xlabel('Absolute Correlation with Target')
        plt.title('Charge Scales by Importance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_charge_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/04_charge_analysis.png")
        
        return charge_correlations
    
    def feature_selection_analysis(self):
        """Analyze which features should be used for prediction"""
        print("=" * 50)
        print("FEATURE SELECTION ANALYSIS")
        print("=" * 50)
        
        # Calculate correlations
        correlations = self.df[self.numeric_features + [self.target]].corr()[self.target].drop(self.target)
        correlations_abs = correlations.abs().sort_values(ascending=False)
        
        # Statistical tests
        print("\n1. TOP FEATURES BY CORRELATION:")
        print("-" * 30)
        for i, (feature, corr) in enumerate(correlations_abs.head(10).items(), 1):
            direction = "positive" if correlations[feature] > 0 else "negative"
            print(f"{i:2d}. {feature:<25} {corr:.3f} ({direction})")
        
        print("\n2. FEATURE CATEGORIES ANALYSIS:")
        print("-" * 30)
        
        # Amino acid features
        aa_importance = correlations_abs[correlations_abs.index.isin(self.aa_features)]
        print(f"Amino acids - Top 5:")
        for feature, corr in aa_importance.head(5).items():
            direction = "positive" if correlations[feature] > 0 else "negative"
            print(f"  {feature:<15} {corr:.3f} ({direction})")
        
        # Charge features
        charge_importance = correlations_abs[correlations_abs.index.isin(self.charge_features)]
        print(f"\nCharge scales - Top 5:")
        for feature, corr in charge_importance.head(5).items():
            direction = "positive" if correlations[feature] > 0 else "negative"
            print(f"  {feature:<20} {corr:.3f} ({direction})")
        
        # Other features
        other_importance = correlations_abs[correlations_abs.index.isin(self.other_features)]
        print(f"\nOther features:")
        for feature, corr in other_importance.items():
            direction = "positive" if correlations[feature] > 0 else "negative"
            print(f"  {feature:<20} {corr:.3f} ({direction})")
        
        print("\n3. RECOMMENDED FEATURES FOR PREDICTION:")
        print("-" * 30)
        
        # Select diverse features with high correlation
        recommended = []
        
        # Top overall features
        recommended.extend(correlations_abs.head(5).index.tolist())
        
        # Ensure representation from each category
        if not any(f in self.charge_features for f in recommended):
            recommended.append(charge_importance.index[0])
        
        if not any(f in self.aa_features for f in recommended):
            recommended.append(aa_importance.index[0])
        
        if not any(f in self.other_features for f in recommended):
            recommended.append(other_importance.index[0])
        
        # Remove duplicates while preserving order
        recommended = list(dict.fromkeys(recommended))
        
        print("Recommended feature set:")
        for i, feature in enumerate(recommended, 1):
            corr = correlations_abs[feature]
            direction = "positive" if correlations[feature] > 0 else "negative"
            category = "AA" if feature in self.aa_features else "Charge" if feature in self.charge_features else "Other"
            print(f"{i:2d}. {feature:<25} {corr:.3f} ({direction}) [{category}]")
        
        # Save feature selection summary to text file
        with open(f'{self.output_dir}/feature_selection_summary.txt', 'w') as f:
            f.write("FEATURE SELECTION ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("TOP FEATURES BY CORRELATION:\n")
            f.write("-" * 30 + "\n")
            for i, (feature, corr) in enumerate(correlations_abs.head(10).items(), 1):
                direction = "positive" if correlations[feature] > 0 else "negative"
                f.write(f"{i:2d}. {feature:<25} {corr:.3f} ({direction})\n")
            
            f.write("\nRECOMMENDED FEATURES FOR PREDICTION:\n")
            f.write("-" * 30 + "\n")
            for i, feature in enumerate(recommended, 1):
                corr = correlations_abs[feature]
                direction = "positive" if correlations[feature] > 0 else "negative"
                category = "AA" if feature in self.aa_features else "Charge" if feature in self.charge_features else "Other"
                f.write(f"{i:2d}. {feature:<25} {corr:.3f} ({direction}) [{category}]\n")
        
        print(f"\nSaved: {self.output_dir}/feature_selection_summary.txt")
        
        return recommended, correlations_abs
    
    def run_complete_analysis(self):
        """Run the complete exploratory data analysis"""
        print("Starting Data Exploration...")
        print("=" * 50)
       
        print("\n1. Analyzing target distribution...")
        self.plot_target_distribution()
        
        print("\n2. Analyzing amino acid features...")
        self.plot_numeric_features(self.aa_features, "AA - ", "02_amino_acids")
        aa_correlations = self.analyze_amino_acid_patterns()
        
        print("\n3. Analyzing charge features...")
        self.plot_numeric_features(self.charge_features, "Charge - ", "02_charge_features")
        charge_correlations = self.analyze_charge_features()
        
        print("\n4. Analyzing other features...")
        if self.other_features:
            self.plot_numeric_features(self.other_features, "", "02_other_features")
        
        print("\n5. Performing correlation analysis...")
        correlations_abs = self.plot_correlation_analysis()
        
        print("\n6. Analyzing feature relationships...")
        self.plot_feature_relationships()
        
        print("\n7. Feature selection analysis...")
        recommended_features, all_correlations = self.feature_selection_analysis()
        
        print(f"\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"All plots saved in: {self.output_dir}/")
        print(f"Recommended features: {len(recommended_features)}")
        print("Check the saved plots and summary file for detailed insights!")
        
        return {
            'recommended_features': recommended_features,
            'all_correlations': all_correlations,
            'aa_correlations': aa_correlations,
            'charge_correlations': charge_correlations
        }

if __name__ == "__main__":
    explorer = DataExplorer('data/IPC_classification_features_train.csv', output_dir='results/features')
    
    results = explorer.run_complete_analysis()
