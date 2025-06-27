import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

"""
Task 2.4: DeepProteinNet with Different Architectures
Testing various configurations of deep neural networks
"""

class ProteinDataset(Dataset):
    """Dataset for protein features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DeepProteinNet(nn.Module):
    """Configurable Deep Neural Network for Protein Classification"""
    
    def __init__(self, input_dim, config):
        super(DeepProteinNet, self).__init__()
        
        self.config = config
        layers = []
        
        # Input dimension
        current_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(config['hidden_dims']):
            # Linear layer
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Activation function
            if config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif config['activation'] == 'elu':
                layers.append(nn.ELU())
            elif config['activation'] == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif config['activation'] == 'gelu':
                layers.append(nn.GELU())
            
            # Batch normalization
            if config['use_batch_norm']:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout
            if config['dropout_rate'] > 0:
                layers.append(nn.Dropout(config['dropout_rate']))
            
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layers': len(self.config['hidden_dims']) + 1,
            'config': self.config
        }

class DeepProteinTrainer:
    """Trainer for different DeepProteinNet configurations"""
    
    def __init__(self, train_file, val_file=None, test_file=None, output_dir='results/dl_model'):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.output_dir = output_dir
        
        Path(self.output_dir).mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.load_data()
        self.define_architectures()
    
    def load_data(self):
        """Load and prepare data"""
        print("=== LOADING DATA ===")
        
        # Load training data
        train_df = pd.read_csv(self.train_file)

        # Prepare features and target
        columns_to_drop = ['uid', 'data_type', 'pI']
        target_col = 'label'
        
        X_train = train_df.drop(columns_to_drop + [target_col], axis=1, errors='ignore')
        X_train = X_train.select_dtypes(include=[np.number])
        y_train = train_df[target_col]
        
        # Load validation data
        if self.val_file:
            val_df = pd.read_csv(self.val_file)
            X_val = val_df.drop(columns_to_drop + [target_col], axis=1, errors='ignore')
            X_val = X_val.select_dtypes(include=[np.number])
            y_val = val_df[target_col]
        else:
            X_val, y_val = None, None
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train.values
        
        if X_val is not None:
            self.X_val = self.scaler.transform(X_val)
            self.y_val = y_val.values
        else:
            self.X_val, self.y_val = None, None
        
        self.input_dim = self.X_train.shape[1]
        print(f"Input dimension: {self.input_dim}")
        print(f"Training samples: {len(self.X_train)}")
        # Save scaler
        scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save feature column names
        feature_cols_path = os.path.join(self.output_dir, 'feature_columns.json')
        with open(feature_cols_path, 'w') as f:
            json.dump(list(X_train.columns), f)

    
    def define_architectures(self):
        """Define different architecture configurations"""
        self.architectures = {
            'Deep': {
                'hidden_dims': [256, 128, 64, 32],
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.3
            },
            'Very_Deep': {
                'hidden_dims': [512, 256, 128, 64, 32],
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.4
            },
            'Wide': {
                'hidden_dims': [512, 256],
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.3
            },
            'ELU_Deep': {
                'hidden_dims': [256, 128, 64, 32],
                'activation': 'elu',
                'use_batch_norm': True,
                'dropout_rate': 0.3
            },
            'Shallow': {
                'hidden_dims': [64, 32],
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.2
            },
            'Medium': {
                'hidden_dims': [128, 64, 32],
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.3
            },
            'GELU_Medium': {
                'hidden_dims': [128, 64, 32],
                'activation': 'gelu',
                'use_batch_norm': True,
                'dropout_rate': 0.2
            },
            'No_BatchNorm': {
                'hidden_dims': [128, 64, 32],
                'activation': 'relu',
                'use_batch_norm': False,
                'dropout_rate': 0.4
            },
            'High_Dropout': {
                'hidden_dims': [256, 128, 64],
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.5
            },
            'Ultra_Deep': {
                'hidden_dims': [256, 256, 128, 128, 64, 64, 32],
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.3
            }
        }
    
    def create_data_loaders(self, batch_size=32):
        """Create data loaders"""
        train_dataset = ProteinDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if self.X_val is not None:
            val_dataset = ProteinDataset(self.X_val, self.y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, model, train_loader, val_loader, epochs=300):
        """Train a single model"""
        model = model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features).view(-1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            if val_loader:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(self.device), labels.to(self.device)
                        
                        outputs = model(features).squeeze()
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        predicted = (outputs > 0.5).float()
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                scheduler.step(val_loss)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
        
        return {
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs,
            'best_val_acc': best_val_acc if val_loader else train_accs[-1]
        }
    
    def compare_architectures(self):
        """Compare all architecture configurations"""
        
        train_loader, val_loader = self.create_data_loaders()
        results = {}
        
        for name, config in self.architectures.items():
            print(f"\nTraining {name} architecture...")
            print(f"Config: {config}")
            
            # Create model
            model = DeepProteinNet(self.input_dim, config)
            model_info = model.get_info()
            
            print(f"Parameters: {model_info['total_params']:,}")
            print(f"Layers: {model_info['layers']}")
            
            # Train model
            history = self.train_model(model, train_loader, val_loader, epochs=50)
            
            # Store results
            results[name] = {
                'model': model,
                'history': history,
                'config': config,
                'model_info': model_info,
                'final_acc': history['best_val_acc']
            }
            
            print(f"{name} final accuracy: {history['best_val_acc']:.4f}")
            
            torch.save(model.state_dict(), f'{self.output_dir}/{name}_model.pth')
        
        return results
    
    def analyze_results(self, results):
        """Analyze and visualize results"""
        print("\n=== ANALYZING RESULTS ===")
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in results.items():
            info = result['model_info']
            comparison_data.append({
                'Architecture': name,
                'Layers': info['layers'],
                'Parameters': info['total_params'],
                'Accuracy': result['final_acc'],
                'Activation': result['config']['activation'],
                'BatchNorm': result['config']['use_batch_norm'],
                'Dropout': result['config']['dropout_rate']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False)
        
        print("\nArchitecture Comparison (sorted by accuracy):")
        print(df.to_string(index=False))
        
        # Save results
        df.to_csv(f'{self.output_dir}/architecture_comparison.csv', index=False)
        
        return df
    
    def plot_results(self, results):
        """Plot comprehensive results"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training curves for top 4 models
        top_4 = sorted(results.items(), key=lambda x: x[1]['final_acc'], reverse=True)[:4]
        
        for i, (name, result) in enumerate(top_4):
            plt.subplot(3, 4, i+1)
            history = result['history']
            
            plt.plot(history['train_acc'], label='Train', linewidth=2)
            if history['val_acc']:
                plt.plot(history['val_acc'], label='Validation', linewidth=2)
            
            plt.title(f'{name}\nAcc: {result["final_acc"]:.3f}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Accuracy comparison
        plt.subplot(3, 2, 3)
        names = list(results.keys())
        accuracies = [results[name]['final_acc'] for name in names]
        
        bars = plt.bar(range(len(names)), accuracies, color='skyblue', alpha=0.7)
        plt.xlabel('Architecture')
        plt.ylabel('Final Accuracy')
        plt.title('Architecture Performance Comparison')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Parameters vs Accuracy
        plt.subplot(3, 2, 4)
        params = [results[name]['model_info']['total_params'] for name in names]
        
        plt.scatter(params, accuracies, s=100, alpha=0.7, c='red')
        for i, name in enumerate(names):
            plt.annotate(name, (params[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Number of Parameters')
        plt.ylabel('Final Accuracy')
        plt.title('Parameters vs Performance')
        plt.grid(True, alpha=0.3)
        
        # 4. Depth vs Accuracy
        plt.subplot(3, 2, 5)
        depths = [results[name]['model_info']['layers'] for name in names]
        
        plt.scatter(depths, accuracies, s=100, alpha=0.7, c='green')
        for i, name in enumerate(names):
            plt.annotate(name, (depths[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Number of Layers')
        plt.ylabel('Final Accuracy')
        plt.title('Depth vs Performance')
        plt.grid(True, alpha=0.3)
        
        # 5. Activation function comparison
        plt.subplot(3, 2, 6)
        activation_acc = {}
        for name, result in results.items():
            activation = result['config']['activation']
            if activation not in activation_acc:
                activation_acc[activation] = []
            activation_acc[activation].append(result['final_acc'])
        
        activations = list(activation_acc.keys())
        avg_accs = [np.mean(activation_acc[act]) for act in activations]
        
        plt.bar(activations, avg_accs, color='orange', alpha=0.7)
        plt.xlabel('Activation Function')
        plt.ylabel('Average Accuracy')
        plt.title('Activation Function Performance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_best_model(self, results):
        """Save the best performing model"""
        best_name = max(results.keys(), key=lambda k: results[k]['final_acc'])
        best_result = results[best_name]
        
        print(f"\n=== BEST MODEL ===")
        print(f"Architecture: {best_name}")
        print(f"Accuracy: {best_result['final_acc']:.4f}")
        print(f"Configuration: {best_result['config']}")
        print(f"Parameters: {best_result['model_info']['total_params']:,}")
        
        # Save best model info
        import json
        best_info = {
            'name': best_name,
            'accuracy': best_result['final_acc'],
            'config': best_result['config'],
            'model_info': best_result['model_info']
        }
        
        with open(f'{self.output_dir}/best_model_info.json', 'w') as f:
            json.dump(best_info, f, indent=2)
    
    def plot_results(self, results):
        """Plot comprehensive results for ALL DeepProteinNet architectures with fixed layout"""

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        n_models = len(results)
        cols = 4
        rows = (n_models + cols - 1) // cols
                
        # FIGURE 1: Training curves for all models
        fig1, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        print(f"Creating training curves for all {n_models} models...")
        
        for i, (name, result) in enumerate(results.items()):
            ax = axes[i]
            history = result['history']
            
            epochs = range(1, len(history['train_acc']) + 1)
            ax.plot(epochs, history['train_acc'], label='Train', linewidth=2, color='blue')
            if history['val_acc']:
                ax.plot(epochs, history['val_acc'], label='Validation', linewidth=2, color='red')
            
            ax.set_title(f'{name}\nAcc: {result["final_acc"]:.3f}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # Hide unused subplots
        for j in range(len(results), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_curves_all_models.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # FIGURE 2: Comprehensive analysis plots
        fig2 = plt.figure(figsize=(20, 15))
        
        names = list(results.keys())
        accuracies = [results[name]['final_acc'] for name in names]
        params = [results[name]['model_info']['total_params'] for name in names]
        depths = [results[name]['model_info']['layers'] for name in names]
        
        # 1. Accuracy comparison bar chart
        plt.subplot(3, 3, 1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = plt.bar(range(len(names)), accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        plt.xlabel('Architecture')
        plt.ylabel('Final Accuracy')
        plt.title('ALL Architectures Performance Comparison')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 2. Parameters vs Accuracy scatter plot
        plt.subplot(3, 3, 2)
        scatter = plt.scatter(params, accuracies, s=120, alpha=0.7, c=accuracies, 
                            cmap='viridis', edgecolors='black', linewidth=1)
        
        for i, name in enumerate(names):
            plt.annotate(name, (params[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.xlabel('Number of Parameters')
        plt.ylabel('Final Accuracy')
        plt.title('Model Complexity vs Performance')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Accuracy')
        
        # 3. Depth vs Accuracy scatter plot
        plt.subplot(3, 3, 3)
        scatter2 = plt.scatter(depths, accuracies, s=120, alpha=0.7, c='green', 
                            edgecolors='black', linewidth=1)
        
        for i, name in enumerate(names):
            plt.annotate(name, (depths[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.xlabel('Number of Layers')
        plt.ylabel('Final Accuracy')
        plt.title('Network Depth vs Performance')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(set(depths)) > 1:  # Only if we have variation in depths
            z = np.polyfit(depths, accuracies, 1)
            p = np.poly1d(z)
            plt.plot(sorted(depths), p(sorted(depths)), "r--", alpha=0.8, linewidth=2)
        
        # 4. Activation function comparison
        plt.subplot(3, 3, 4)
        activation_acc = {}
        activation_counts = {}
        
        for name, result in results.items():
            activation = result['config']['activation']
            if activation not in activation_acc:
                activation_acc[activation] = []
                activation_counts[activation] = 0
            activation_acc[activation].append(result['final_acc'])
            activation_counts[activation] += 1
        
        activations = list(activation_acc.keys())
        avg_accs = [np.mean(activation_acc[act]) for act in activations]
        std_accs = [np.std(activation_acc[act]) if len(activation_acc[act]) > 1 else 0 for act in activations]
        
        bars = plt.bar(activations, avg_accs, color='orange', alpha=0.7, 
                    edgecolor='black', linewidth=1)
        
        # Add error bars if multiple samples
        plt.errorbar(activations, avg_accs, yerr=std_accs, fmt='none', 
                    color='black', capsize=5, capthick=2)
        
        plt.xlabel('Activation Function')
        plt.ylabel('Average Accuracy')
        plt.title('Activation Function Performance')
        plt.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, avg_acc, count) in enumerate(zip(bars, avg_accs, [activation_counts[act] for act in activations])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{avg_acc:.3f}\n(n={count})', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        # 5. Model ranking (all models)
        plt.subplot(3, 3, 5)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['final_acc'], reverse=True)
        sorted_names = [name for name, _ in sorted_results]
        sorted_accs = [result['final_acc'] for _, result in sorted_results]
        
        y_pos = np.arange(len(sorted_names))
        bars = plt.barh(y_pos, sorted_accs, alpha=0.7, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(sorted_names))))
        
        plt.yticks(y_pos, sorted_names)
        plt.xlabel('Final Accuracy')
        plt.title('Complete Architecture Ranking')
        plt.grid(True, alpha=0.3, axis='x')
        
        for bar, acc in zip(bars, sorted_accs):
            plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.3f}', va='center', ha='left', fontsize=8)
        
        # 6. Parameter efficiency
        plt.subplot(3, 3, 6)
        df = pd.DataFrame({
            'Architecture': names,
            'Accuracy': accuracies,
            'Parameters': params
        })
        df['Efficiency'] = df['Accuracy'] / (df['Parameters'] / 1000)
        df_eff = df.sort_values('Efficiency', ascending=True)
        
        bars = plt.barh(range(len(df_eff)), df_eff['Efficiency'], 
                    alpha=0.7, color='purple')
        plt.yticks(range(len(df_eff)), df_eff['Architecture'])
        plt.xlabel('Accuracy per 1K Parameters')
        plt.title('Parameter Efficiency')
        plt.grid(True, alpha=0.3, axis='x')
        
        # 7. Batch Normalization effect
        plt.subplot(3, 3, 7)
        bn_groups = {}
        for name, result in results.items():
            bn_status = result['config']['use_batch_norm']
            if bn_status not in bn_groups:
                bn_groups[bn_status] = []
            bn_groups[bn_status].append(result['final_acc'])
        
        bn_labels = ['Without BatchNorm', 'With BatchNorm']
        bn_means = [np.mean(bn_groups.get(False, [0])), np.mean(bn_groups.get(True, [0]))]
        bn_stds = [np.std(bn_groups.get(False, [0])), np.std(bn_groups.get(True, [0]))]
        
        bars = plt.bar(bn_labels, bn_means, yerr=bn_stds, capsize=5, 
                    alpha=0.7, color=['lightcoral', 'lightgreen'])
        plt.ylabel('Average Accuracy')
        plt.title('Batch Normalization Effect')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, mean in zip(bars, bn_means):
            if mean > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 8. Dropout rate analysis
        plt.subplot(3, 3, 8)
        dropout_groups = {}
        for name, result in results.items():
            dropout_rate = result['config']['dropout_rate']
            if dropout_rate not in dropout_groups:
                dropout_groups[dropout_rate] = []
            dropout_groups[dropout_rate].append(result['final_acc'])
        
        dropout_rates = sorted(dropout_groups.keys())
        dropout_means = [np.mean(dropout_groups[rate]) for rate in dropout_rates]
        
        plt.bar([str(rate) for rate in dropout_rates], dropout_means, 
                alpha=0.7, color='cyan')
        plt.xlabel('Dropout Rate')
        plt.ylabel('Average Accuracy')
        plt.title('Dropout Rate vs Performance')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 9. Summary statistics
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        stats_text = f"""
    SUMMARY STATISTICS

    Total Models: {len(results)}
    Best Accuracy: {max(accuracies):.3f}
    Worst Accuracy: {min(accuracies):.3f}
    Mean Accuracy: {np.mean(accuracies):.3f}
    Std Deviation: {np.std(accuracies):.3f}

    Parameter Range:
    Min: {min(params):,}
    Max: {max(params):,}
    Mean: {int(np.mean(params)):,}

    Best Model: {sorted_names[0]}
    Most Efficient: {df_eff.iloc[-1]['Architecture']}
        """
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comprehensive_analysis_all_models.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Created comprehensive analysis for all {n_models} models")

        
    def plot_all_training_curves(self, results):
        """Create separate detailed training curves for models"""        
        n_models = len(results)
        cols = 4
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        # Handle single row case
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        print(f"Creating detailed training curves for all {n_models} models...")
        
        for i, (name, result) in enumerate(results.items()):
            ax = axes[i]
            history = result['history']
            
            epochs = range(1, len(history['train_acc']) + 1)
            
            # Plot accuracy
            ax.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            if history['val_acc']:
                ax.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            
            # Create secondary y-axis for loss
            ax2 = ax.twinx()
            ax2.plot(epochs, history['train_loss'], 'b--', alpha=0.6, label='Training Loss', linewidth=1)
            if history['val_loss']:
                ax2.plot(epochs, history['val_loss'], 'r--', alpha=0.6, label='Validation Loss', linewidth=1)
            
            # Formatting
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy', color='black')
            ax2.set_ylabel('Loss', color='gray')
            ax.set_title(f'{name}\nFinal Acc: {result["final_acc"]:.3f}\nParams: {result["model_info"]["total_params"]:,}', 
                        fontsize=10, fontweight='bold')
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
            
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # Hide unused subplots
        for j in range(len(results), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_training_curves_detailed.png', dpi=300, bbox_inches='tight')

    
    def run_analysis(self):
        """Run complete architecture comparison"""
        print("="*70)
        

        # Compare architectures
        results = self.compare_architectures()
        
        # Analyze results
        comparison_df = self.analyze_results(results)
        
        # Create plots for ALL models
        print(f"\nGenerating plots for ALL {len(results)} models...")
        
        # 1. Comprehensive plot with all models
        self.plot_results(results)
        
        # 2. Detailed training curves for all models
        self.plot_all_training_curves(results)
        
        # 3. Architecture analysis (if you have this function)
        if hasattr(self, 'plot_architecture_analysis'):
            self.plot_architecture_analysis(results)
        
        # 4. Summary report (if you have this function)
        if hasattr(self, 'create_summary_report'):
            self.create_summary_report(results, comparison_df)
        
        # Save best model
        self.save_best_model(results)
        
        print(f"\nAll plots saved in: {self.output_dir}/")
        print(f"Total models analyzed: {len(results)}")
        
        return results, comparison_df

def main():
    """Main function"""
    train_file = 'data/IPC_classification_features_train.csv'
    val_file = 'data/IPC_classification_features_val.csv'
    test_file = 'data/IPC_classification_features_test.csv'
    
    trainer = DeepProteinTrainer(train_file, val_file, test_file)
    results, comparison_df = trainer.run_analysis()
    
    return results, comparison_df

if __name__ == "__main__":
    results, comparison_df = main()


