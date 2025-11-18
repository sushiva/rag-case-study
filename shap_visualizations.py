"""
SHAP Visualization Module for Energy Forecasting
Provides comprehensive feature effect visualizations using SHAP values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path

class SHAPVisualizer:
    """
    Creates beautiful visualizations of feature effects using SHAP values
    """
    
    def __init__(self, model, X_train, X_test, feature_names=None):
        """
        Initialize SHAP visualizer
        
        Parameters:
        -----------
        model : trained model
            Model to explain (XGBoost, Random Forest, or Neural Network)
        X_train : array-like
            Training features for background data
        X_test : array-like
            Test features to explain
        feature_names : list, optional
            Names of features for better visualization
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names if feature_names else [f"Feature {i}" for i in range(X_train.shape[1])]
        
        # Create SHAP explainer
        self.explainer = None
        self.shap_values = None
        self._create_explainer()
        
    def _create_explainer(self):
        """Create appropriate SHAP explainer based on model type"""
        model_name = type(self.model).__name__.lower()
        
        print(f"Creating SHAP explainer for {model_name}...")
        
        if 'xgboost' in model_name or 'xgb' in model_name:
            # Tree explainer for XGBoost
            self.explainer = shap.TreeExplainer(self.model)
            
        elif 'random' in model_name or 'forest' in model_name:
            # Tree explainer for Random Forest
            self.explainer = shap.TreeExplainer(self.model)
            
        else:
            # Use Kernel explainer for other models (e.g., Neural Networks)
            # Use a sample of training data as background
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
        
        print("✓ SHAP explainer created successfully")
    
    def calculate_shap_values(self, sample_size=None):
        """
        Calculate SHAP values for test set
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of samples to explain (for faster computation)
        """
        print("\nCalculating SHAP values...")
        
        if sample_size and sample_size < len(self.X_test):
            X_explain = self.X_test[:sample_size]
        else:
            X_explain = self.X_test
        
        self.shap_values = self.explainer.shap_values(X_explain)
        print(f"✓ SHAP values calculated for {len(X_explain)} samples")
        
        return self.shap_values
    
    def plot_summary(self, save_path=None, max_display=15):
        """
        Create SHAP summary plot showing feature importance
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        max_display : int
            Maximum number of features to display
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_test[:len(self.shap_values)],
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("Feature Impact on Energy Predictions\n(SHAP Summary)", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_bar_importance(self, save_path=None, max_display=15):
        """
        Create bar plot of mean absolute SHAP values
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        max_display : int
            Maximum number of features to display
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_test[:len(self.shap_values)],
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title("Feature Importance (Mean |SHAP Value|)", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Bar importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_dependence(self, feature_name, interaction_feature='auto', save_path=None):
        """
        Create SHAP dependence plot for a specific feature
        
        Parameters:
        -----------
        feature_name : str
            Name of feature to plot
        interaction_feature : str or int
            Feature to use for coloring ('auto' for automatic selection)
        save_path : str, optional
            Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Find feature index
        if feature_name not in self.feature_names:
            print(f"Feature '{feature_name}' not found")
            return
        
        feature_idx = self.feature_names.index(feature_name)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            self.X_test[:len(self.shap_values)],
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f"SHAP Dependence: {feature_name}", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dependence plot saved to {save_path}")
        
        plt.show()
    
    def plot_force_single(self, instance_idx=0, matplotlib=True, save_path=None):
        """
        Create force plot for a single prediction
        
        Parameters:
        -----------
        instance_idx : int
            Index of instance to explain
        matplotlib : bool
            Use matplotlib rendering (True) or interactive HTML (False)
        save_path : str, optional
            Path to save the plot (only for matplotlib=True)
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        if matplotlib:
            plt.figure(figsize=(14, 4))
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[instance_idx],
                self.X_test[instance_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f"Force Plot: Prediction Explanation (Sample {instance_idx})", 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Force plot saved to {save_path}")
            
            plt.show()
        else:
            # Interactive HTML visualization
            return shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[instance_idx],
                self.X_test[instance_idx],
                feature_names=self.feature_names
            )
    
    def plot_waterfall(self, instance_idx=0, save_path=None):
        """
        Create waterfall plot showing how features contribute to prediction
        
        Parameters:
        -----------
        instance_idx : int
            Index of instance to explain
        save_path : str, optional
            Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(10, 8))
        
        # Create explanation object for waterfall plot
        explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=self.explainer.expected_value,
            data=self.X_test[instance_idx],
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"Waterfall Plot: Feature Contributions (Sample {instance_idx})", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Waterfall plot saved to {save_path}")
        
        plt.show()
    
    def plot_decision(self, instance_idx=0, save_path=None):
        """
        Create decision plot showing cumulative feature effects
        
        Parameters:
        -----------
        instance_idx : int
            Index of instance to explain
        save_path : str, optional
            Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(10, 8))
        shap.decision_plot(
            self.explainer.expected_value,
            self.shap_values[instance_idx],
            self.X_test[instance_idx],
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f"Decision Plot: Cumulative Feature Effects (Sample {instance_idx})", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Decision plot saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, output_dir='shap_visualizations', 
                                   top_features=5, sample_indices=[0, 1, 2]):
        """
        Generate a comprehensive set of SHAP visualizations
        
        Parameters:
        -----------
        output_dir : str
            Directory to save all plots
        top_features : int
            Number of top features to create dependence plots for
        sample_indices : list
            Indices of samples to create individual explanation plots for
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE SHAP VISUALIZATION REPORT")
        print("="*60)
        
        # Calculate SHAP values once
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # 1. Summary plot
        print("\n1. Creating summary plot...")
        self.plot_summary(save_path=output_path / 'summary_plot.png')
        
        # 2. Bar importance plot
        print("\n2. Creating bar importance plot...")
        self.plot_bar_importance(save_path=output_path / 'importance_bar.png')
        
        # 3. Top feature dependence plots
        print(f"\n3. Creating dependence plots for top {top_features} features...")
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        top_feature_indices = np.argsort(mean_abs_shap)[-top_features:][::-1]
        
        for idx in top_feature_indices:
            feature_name = self.feature_names[idx]
            safe_name = feature_name.replace(' ', '_').replace('/', '_')
            print(f"   - {feature_name}")
            self.plot_dependence(
                feature_name, 
                save_path=output_path / f'dependence_{safe_name}.png'
            )
        
        # 4. Individual prediction explanations
        print(f"\n4. Creating individual prediction explanations...")
        for idx in sample_indices:
            if idx < len(self.shap_values):
                print(f"   - Sample {idx}")
                self.plot_waterfall(idx, save_path=output_path / f'waterfall_sample_{idx}.png')
                self.plot_force_single(idx, save_path=output_path / f'force_sample_{idx}.png')
        
        print("\n" + "="*60)
        print(f"✓ COMPREHENSIVE REPORT COMPLETE")
        print(f"✓ All visualizations saved to: {output_path}")
        print("="*60)
        
        return output_path
    
    def get_feature_importance_df(self):
        """
        Get feature importance as a DataFrame
        
        Returns:
        --------
        pd.DataFrame
            Features ranked by mean absolute SHAP value
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Abs_SHAP': mean_abs_shap,
            'Importance_Rank': range(1, len(self.feature_names) + 1)
        })
        
        importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
        importance_df['Importance_Rank'] = range(1, len(importance_df) + 1)
        
        return importance_df


def demo_shap_visualizations():
    """
    Demo function showing how to use the SHAP visualizer
    """
    print("SHAP Visualization Demo")
    print("="*60)
    print("\nThis module provides comprehensive SHAP visualizations including:")
    print("  1. Summary plots - Overall feature importance")
    print("  2. Bar plots - Mean absolute SHAP values")
    print("  3. Dependence plots - Feature effects and interactions")
    print("  4. Force plots - Individual prediction explanations")
    print("  5. Waterfall plots - Step-by-step feature contributions")
    print("  6. Decision plots - Cumulative feature effects")
    print("\nSee the usage example in the main script.")


if __name__ == "__main__":
    demo_shap_visualizations()