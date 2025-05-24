import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle # Kept if needed for future, not used in current plots
from matplotlib.colors import LinearSegmentedColormap
# from scipy.spatial.distance import pdist, squareform # Not used, can be removed
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for maximum aesthetic appeal
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl") # Using a seaborn built-in palette

# Custom color palettes (can be expanded or modified)
colors_custom = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D', 
    'grid4': '#FF6B6B',
    'grid8': '#4ECDC4',
    'grid16': '#45B7D1',
    'relu': '#96CEB4',
    'no_relu': '#FFEAA7'
}

def load_and_process_data(csv_path):
    """Load CSV and add derived columns for analysis.
    Expects 'val_acc', 'flops', 'params', 'latency', 'use_relu', 'prune_amt', 'grid_size', 'width_mult' in CSV.
    """
    df = pd.read_csv(csv_path)
    
    # Check for essential columns
    required_cols = ['flops', 'params', 'latency', 'val_acc', 'use_relu', 'prune_amt', 'grid_size', 'width_mult']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")

    # Add efficiency metrics
    df['flops_millions'] = df['flops'] / 1e6
    df['params_thousands'] = df['params'] / 1e3
    df['latency_ms'] = df['latency'] * 1000
    df['accuracy_pct'] = df['val_acc'] * 100 # Derived from val_acc
    
    # Configuration labels
    df['config'] = df.apply(lambda x: f"{'ReLU' if x['use_relu'] else 'Identity'}_{'Pruned' if x['prune_amt'] > 0 else 'Full'}", axis=1)
    df['grid_label'] = df['grid_size'].apply(lambda x: f'Grid {x}')
    df['width_label'] = df['width_mult'].apply(lambda x: f'{x}x Width')
    
    return df

# --- Helper functions for individual plot components ---

def plot_component_grid_size_impact(df, ax):
    """Plots Grid Size Impact on Accuracy on a given Axes object."""
    if df.empty or 'grid_size' not in df.columns or 'accuracy_pct' not in df.columns:
        ax.text(0.5, 0.5, "Data unavailable for Grid Size Impact", ha='center', va='center')
        ax.set_title('Grid Size Impact on Accuracy (No Data)', fontsize=16, fontweight='bold', pad=20)
        return
        
    grid_means = df.groupby('grid_size')['accuracy_pct'].agg(['mean', 'std']).reset_index()
    
    bar_colors = [colors_custom['grid4'], colors_custom['grid8'], colors_custom['grid16']]
    
    for i, (_, row) in enumerate(grid_means.iterrows()):
        color = bar_colors[i % len(bar_colors)] 
        ax.bar(row['grid_size'], row['mean'], yerr=row['std'], 
               color=color, alpha=0.8, capsize=8, width=2.5,
               edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Grid Size (Spline Knots)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Grid Size Impact on Accuracy', fontsize=16, fontweight='bold', pad=20)
    if not grid_means.empty: # Set ylim based on data if available
        min_y = (grid_means['mean'] - grid_means['std']).min()
        max_y = (grid_means['mean'] + grid_means['std']).max()
        ax.set_ylim(max(0, min_y - 0.5) , min(100, max_y + 0.5)) # Ensure reasonable limits
    else:
        ax.set_ylim(98,100) # Default if no data
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    for i, (_, row) in enumerate(grid_means.iterrows()):
        ax.text(row['grid_size'], row['mean'] + row['std'] + 0.05, 
                f'{row["mean"]:.2f}%', ha='center', va='bottom', fontweight='bold')

def plot_component_efficiency_frontier_basic(df, ax):
    """Plots a basic Efficiency Frontier on a given Axes object."""
    if df.empty or not all(col in df.columns for col in ['params_thousands', 'flops_millions', 'accuracy_pct', 'grid_size']):
        ax.text(0.5, 0.5, "Data unavailable for Efficiency Frontier", ha='center', va='center')
        ax.set_title('Efficiency Frontier (No Data)', fontsize=16, fontweight='bold', pad=20)
        return

    sizes = (df['params_thousands'] / (df['params_thousands'].max() if df['params_thousands'].max() > 0 else 1)) * 300 + 50
    
    scatter = ax.scatter(df['flops_millions'], df['accuracy_pct'], 
                         c=df['grid_size'], s=sizes, alpha=0.7,
                         cmap='viridis', edgecolors='white', linewidth=2)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Grid Size', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('FLOPs (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Efficiency Frontier', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    if len(df['flops_millions'].dropna()) > 1 and len(df['accuracy_pct'].dropna()) > 1 : 
        valid_data = df[['flops_millions', 'accuracy_pct']].dropna()
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['flops_millions'], valid_data['accuracy_pct'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(valid_data['flops_millions'].min(), valid_data['flops_millions'].max(), 100)
            ax.plot(x_trend, p(x_trend), "--", 
                    color=colors_custom['accent'], alpha=0.8, linewidth=2)

def plot_component_performance_heatmap_basic(df, ax):
    """Plots a basic Performance Heatmap on a given Axes object."""
    if df.empty or not all(col in df.columns for col in ['accuracy_pct', 'use_relu', 'prune_amt', 'grid_size']):
        ax.text(0.5, 0.5, "Data unavailable for Performance Heatmap", ha='center', va='center')
        ax.set_title('Performance Heatmap (No Data)', fontsize=16, fontweight='bold', pad=20)
        return
        
    pivot_data = df.pivot_table(values='accuracy_pct', 
                                index=['use_relu', 'prune_amt'], 
                                columns='grid_size', 
                                aggfunc='mean')
    if pivot_data.empty:
        ax.text(0.5, 0.5, "Not enough data diversity for Pivot Table", ha='center', va='center')
        ax.set_title('Performance Heatmap (Pivot Empty)', fontsize=16, fontweight='bold', pad=20)
        return

    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    cmap = LinearSegmentedColormap.from_list('custom_basic_heatmap', colors_list, N=100)
    cmap.set_bad(color='lightgrey') # Set color for NaN values

    # Determine vmin and vmax from the data, handling cases where all values might be NaN
    valid_pivot_values_basic = pivot_data.values[~np.isnan(pivot_data.values)]
    vmin_basic = valid_pivot_values_basic.min() if len(valid_pivot_values_basic) > 0 else 0
    vmax_basic = valid_pivot_values_basic.max() if len(valid_pivot_values_basic) > 0 else 100
    if vmin_basic == vmax_basic: # Avoid error if all valid values are the same
        vmin_basic -= 0.1 
        vmax_basic += 0.1


    im = ax.imshow(pivot_data.values, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin_basic, vmax=vmax_basic)
    
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([f'Grid {x}' for x in pivot_data.columns], fontweight='bold')
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels([f'{"ReLU" if x[0] else "Identity"} {"Pruned" if x[1] > 0 else "Full"}' 
                        for x in pivot_data.index], fontweight='bold')
    
    # Adjust text color based on cell brightness
    mean_heatmap_val_basic = np.nanmean(pivot_data.values) if np.sum(~np.isnan(pivot_data.values)) > 0 else (vmin_basic + vmax_basic) / 2

    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.values[i, j]
            if not np.isnan(value):
                text_color = 'black' if value > mean_heatmap_val_basic else 'white' # Simplified logic
                # A more robust way would be to check luminance of the cell color
                # but this is a common heuristic.
                ax.text(j, i, f'{value:.2f}%', ha='center', va='center', 
                        fontweight='bold', color=text_color, fontsize=10) 
    
    ax.set_title('Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='Accuracy (%)', shrink=0.8)

def plot_component_pareto_frontier_bubble(df, ax):
    """Plots Speed vs Accuracy Trade-off (Bubble Chart) on a given Axes object."""
    if df.empty or not all(col in df.columns for col in ['params_thousands', 'config', 'latency_ms', 'accuracy_pct']):
        ax.text(0.5, 0.5, "Data unavailable for Pareto Frontier", ha='center', va='center')
        ax.set_title('Speed vs Accuracy Trade-off (No Data)', fontsize=16, fontweight='bold', pad=20)
        return

    bubble_sizes = (df['params_thousands'] / (df['params_thousands'].max() if df['params_thousands'].max() > 0 else 1)) * 500 + 100
    
    config_colors_map = {
        'ReLU_Full': colors_custom['grid4'], 'ReLU_Pruned': colors_custom['grid8'], 
        'Identity_Full': colors_custom['grid16'], 'Identity_Pruned': colors_custom['accent']
    }
    
    for config_name in df['config'].unique():
        mask = df['config'] == config_name
        ax.scatter(df[mask]['latency_ms'], df[mask]['accuracy_pct'], 
                   s=bubble_sizes[mask], alpha=0.7, 
                   color=config_colors_map.get(config_name, '#000000'), 
                   label=config_name, edgecolors='white', linewidth=2)
    
    ax.set_xlabel('Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Speed vs Accuracy Trade-off (Bubble size = Parameters)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    if not df.empty and df['accuracy_pct'].notna().any() and df['latency_ms'].notna().any():
        best_acc = df.loc[df['accuracy_pct'].idxmax()]
        fastest = df.loc[df['latency_ms'].idxmin()]
        
        ax.annotate(f'Best Acc\n{best_acc["accuracy_pct"]:.2f}%', 
                    xy=(best_acc['latency_ms'], best_acc['accuracy_pct']),
                    xytext=(best_acc['latency_ms'] + 1, best_acc['accuracy_pct'] - 0.3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=9, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.annotate(f'Fastest\n{fastest["latency_ms"]:.2f}ms', 
                    xy=(fastest['latency_ms'], fastest['accuracy_pct']),
                    xytext=(fastest['latency_ms'] + 1, fastest['accuracy_pct'] + 0.3),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

def plot_component_ablation_radar_chart(df, ax):
    """Plots Top Configurations Radar Chart on a given Polar Axes object."""
    required_metrics = ['accuracy_pct', 'latency_ms', 'flops_millions', 'params_thousands']
    if df.empty or not all(col in df.columns for col in required_metrics + ['grid_size', 'width_mult']):
        ax.text(0.5, 0.5, "Data unavailable for Radar Chart", ha='center', va='center', transform=ax.transAxes) # Use transAxes for polar
        ax.set_title('Top Configurations Radar (No Data)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([]) # Clear ticks for empty polar plot
        ax.set_yticks([])
        return

    categories = ['Accuracy', 'Speed', 'Efficiency', 'Compactness']
    scaler = StandardScaler()
    metrics = df[required_metrics].copy()
    
    metrics['latency_ms'] = 1 / (metrics['latency_ms'].replace(0, 1e-9) + 1e-9) 
    metrics['flops_millions'] = 1 / (metrics['flops_millions'].replace(0, 1e-9) + 1e-9)
    metrics['params_thousands'] = 1 / (metrics['params_thousands'].replace(0, 1e-9) + 1e-9)
    
    metrics_norm = scaler.fit_transform(metrics.fillna(0)) # FillNa before scaling
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1] 
    
    top_configs = df.nlargest(3, 'accuracy_pct')
    if top_configs.empty:
        ax.text(0.5, 0.5, "No top configurations to display", ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Top Configurations Radar (No Top Configs)', fontsize=14, fontweight='bold', pad=20)
        return

    radar_colors_list = [colors_custom['primary'], colors_custom['secondary'], colors_custom['accent']]
    
    for i, (original_idx, config_row) in enumerate(top_configs.iterrows()):
        if original_idx in df.index: 
            loc_in_metrics_norm = df.index.get_loc(original_idx)
            if loc_in_metrics_norm < len(metrics_norm):
                values = metrics_norm[loc_in_metrics_norm].tolist()
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                        label=f'Grid {config_row["grid_size"]}, {config_row["width_mult"]}x',
                        color=radar_colors_list[i % len(radar_colors_list)], alpha=0.8)
                ax.fill(angles, values, alpha=0.2, color=radar_colors_list[i % len(radar_colors_list)])
            else:
                print(f"Warning: Index {original_idx} (loc {loc_in_metrics_norm}) for top config is out of bounds for metrics_norm.")
        else:
            print(f"Warning: Original index {original_idx} from top_configs not found in main DataFrame.")


    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_title('Top Configurations Radar', fontsize=14, fontweight='bold', pad=30) 
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=9) 
    ax.grid(True, alpha=0.3)

def plot_component_parameter_scaling_analysis(df, ax):
    """Plots Parameter Scaling Analysis (Stacked Bar Chart) on a given Axes object."""
    if df.empty or not all(col in df.columns for col in ['width_mult', 'grid_size', 'params_thousands']):
        ax.text(0.5, 0.5, "Data unavailable for Parameter Scaling", ha='center', va='center')
        ax.set_title('Parameter Scaling Analysis (No Data)', fontsize=16, fontweight='bold', pad=20)
        return

    width_groups = df.groupby(['width_mult', 'grid_size'])['params_thousands'].mean().unstack()
    if width_groups.empty:
        ax.text(0.5, 0.5, "Not enough data diversity for Parameter Scaling", ha='center', va='center')
        ax.set_title('Parameter Scaling Analysis (Pivot Empty)', fontsize=16, fontweight='bold', pad=20)
        return

    x_labels = [f'{w}x' for w in width_groups.index]
    x_pos = np.arange(len(x_labels))
    bar_width = 0.6 
    
    bottom = np.zeros(len(width_groups.index))
    grid_colors_list = [colors_custom['grid4'], colors_custom['grid8'], colors_custom['grid16']]
    
    for i, (grid_size_val, color) in enumerate(zip(width_groups.columns, grid_colors_list)):
        values = width_groups[grid_size_val].fillna(0) 
        ax.bar(x_pos, values, width=bar_width, bottom=bottom, 
               label=f'Grid {grid_size_val}', color=color, alpha=0.8,
               edgecolor='white', linewidth=1)
        bottom += values
    
    ax.set_xlabel('Width Multiplier', fontsize=14, fontweight='bold')
    ax.set_ylabel('Parameters (Thousands)', fontsize=14, fontweight='bold')
    ax.set_title('Parameter Scaling Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9) 
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f9fa')
    
    for i, (idx, row) in enumerate(width_groups.iterrows()): 
        total = row.sum()
        if total > 0 : 
             ax.text(x_pos[i], total + 5, f'{total:.0f}K', ha='center', va='bottom', fontweight='bold')

def create_stunning_plots(df, output_filepath_prefix):
    """Creates the comprehensive 6-panel plot and saves it."""
    fig = plt.figure(figsize=(20, 18)) 
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], 
                          hspace=0.5, wspace=0.35) 
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_component_grid_size_impact(df, ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_component_efficiency_frontier_basic(df, ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_component_performance_heatmap_basic(df, ax3)
    
    ax4 = fig.add_subplot(gs[1, :])
    plot_component_pareto_frontier_bubble(df, ax4)
    
    ax5 = fig.add_subplot(gs[2, 0], projection='polar')
    plot_component_ablation_radar_chart(df, ax5)
    
    ax6 = fig.add_subplot(gs[2, 1:]) 
    plot_component_parameter_scaling_analysis(df, ax6)
    
    fig.suptitle('FastKAN Ablation Study - Comprehensive Analysis', 
                 fontsize=24, fontweight='bold', y=0.99) 
    
    fig.patch.set_facecolor('#fafafa')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    save_path_png = f'{output_filepath_prefix}_comprehensive.png'
    save_path_pdf = f'{output_filepath_prefix}_comprehensive.pdf'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {save_path_png}")
    print(f"Saved: {save_path_pdf}")
    plt.close(fig)

def save_individual_component_figures(df, output_dir, base_filename_prefix):
    """Saves each component of the comprehensive plot as an individual figure."""
    component_plotters = {
        "grid_impact": (plot_component_grid_size_impact, {}),
        "efficiency_basic": (plot_component_efficiency_frontier_basic, {}),
        "heatmap_basic": (plot_component_performance_heatmap_basic, {}),
        "pareto_frontier": (plot_component_pareto_frontier_bubble, {"figsize": (12,8)}),
        "radar_chart": (plot_component_ablation_radar_chart, {"projection": "polar", "figsize": (8,8)}),
        "parameter_scaling": (plot_component_parameter_scaling_analysis, {"figsize": (10,7)})
    }

    for name, (plot_func, kwargs) in component_plotters.items():
        fig_size = kwargs.get("figsize", (8, 6)) 
        projection = kwargs.get("projection")

        if projection == "polar":
            fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': 'polar'})
        else:
            fig, ax = plt.subplots(figsize=fig_size)
        
        plot_func(df, ax) 
        
        plt.tight_layout()
        filepath_prefix = os.path.join(output_dir, f"{base_filename_prefix}_{name}")
        save_path_png = f'{filepath_prefix}.png'
        save_path_pdf = f'{filepath_prefix}.pdf'
        
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        print(f"Saved: {save_path_png}")
        print(f"Saved: {save_path_pdf}")
        plt.close(fig)

def create_special_individual_plots(df, output_filepath_prefix):
    """Creates and saves the two special individual high-impact plots."""
    if df.empty:
        print("Warning: DataFrame is empty. Skipping special individual plots.")
        return

    # 1. SEXY EFFICIENCY PLOT (Special Version)
    plt.figure(figsize=(12, 8))
    ax_eff = plt.gca()
    
    if not all(col in df.columns for col in ['params_thousands', 'flops_millions', 'accuracy_pct', 'grid_size']):
        ax_eff.text(0.5,0.5, "Data missing for Special Efficiency Plot", ha='center', va='center')
    else:
        sizes = (df['params_thousands'] / (df['params_thousands'].max() if df['params_thousands'].max() > 0 else 1)) * 400 + 100
        scatter = ax_eff.scatter(df['flops_millions'], df['accuracy_pct'], 
                                 c=df['grid_size'], s=sizes, alpha=0.8,
                                 cmap='plasma', edgecolors='white', linewidth=2)
        
        if len(df['flops_millions'].dropna()) > 1 and len(df['accuracy_pct'].dropna()) > 1:
            valid_data = df[['flops_millions', 'accuracy_pct']].dropna()
            if len(valid_data)>1:
                z = np.polyfit(valid_data['flops_millions'], valid_data['accuracy_pct'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(valid_data['flops_millions'].min(), valid_data['flops_millions'].max(), 100)
                ax_eff.plot(x_trend, p(x_trend), "--", color='red', alpha=0.8, linewidth=3, label='Trend')
        
        cbar = plt.colorbar(scatter, ax=ax_eff)
        cbar.set_label('Grid Size (Spline Knots)', fontsize=14, fontweight='bold')
        
        if 'flops_millions' in df.columns and df['flops_millions'].notna().any() and (df['flops_millions'].abs() > 1e-9).any(): 
            efficiency_ratio = df['accuracy_pct'] / (df['flops_millions'].replace(0, 1e-9) + 1e-9) 
            idx_best_efficiency = efficiency_ratio.idxmax()
            best_efficiency_row = df.loc[idx_best_efficiency]
            
            ax_eff.annotate('Best Efficiency', 
                            xy=(best_efficiency_row['flops_millions'], best_efficiency_row['accuracy_pct']),
                            xytext=(best_efficiency_row['flops_millions'] * 1.2 + 0.1, best_efficiency_row['accuracy_pct'] - 0.2), 
                            arrowprops=dict(arrowstyle='->', color='gold', lw=3),
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.8))

    ax_eff.set_xlabel('FLOPs (Millions)', fontsize=16, fontweight='bold')
    ax_eff.set_ylabel('Validation Accuracy (%)', fontsize=16, fontweight='bold')
    ax_eff.set_title('FastKAN Efficiency Frontier (Special)\nBubble size = Parameters', 
                     fontsize=20, fontweight='bold', pad=20)
    ax_eff.grid(True, alpha=0.3)
    ax_eff.set_facecolor('#f8f9fa')
    plt.tight_layout()
    save_path_png_eff = f'{output_filepath_prefix}_efficiency_frontier_special.png'
    save_path_pdf_eff = f'{output_filepath_prefix}_efficiency_frontier_special.pdf'
    plt.savefig(save_path_png_eff, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf_eff, bbox_inches='tight')
    print(f"Saved: {save_path_png_eff}")
    print(f"Saved: {save_path_pdf_eff}")
    plt.close() 
    
    # 2. GORGEOUS HEATMAP (Special Version)
    plt.figure(figsize=(14, 10))
    ax_hm = plt.gca()
    
    if not all(col in df.columns for col in ['accuracy_pct', 'grid_size', 'width_mult', 'use_relu', 'prune_amt']):
        ax_hm.text(0.5,0.5, "Data missing for Special Heatmap", ha='center', va='center')
    else:
        pivot_full = df.pivot_table(values='accuracy_pct', 
                                    index=['grid_size', 'width_mult'], 
                                    columns=['use_relu', 'prune_amt'], 
                                    aggfunc='mean')
        if pivot_full.empty:
            ax_hm.text(0.5,0.5, "Not enough data diversity for Special Heatmap pivot", ha='center', va='center')
        else:
            colors_list_hm = ['#FF416C', '#FF4B2B', '#FF8E53', '#FF6B6B', '#4ECDC4', '#45B7D1']
            cmap_hm = LinearSegmentedColormap.from_list('fastkan_special', colors_list_hm, N=256)
            cmap_hm.set_bad(color='lightgrey') # Set color for NaN values

            valid_pivot_full_values = pivot_full.values[~np.isnan(pivot_full.values)]
            vmin_special = valid_pivot_full_values.min() if len(valid_pivot_full_values) > 0 else 0
            vmax_special = valid_pivot_full_values.max() if len(valid_pivot_full_values) > 0 else 100
            if vmin_special == vmax_special: # Avoid error if all valid values are the same
                vmin_special -= 0.1
                vmax_special += 0.1

            im = ax_hm.imshow(pivot_full.values, cmap=cmap_hm, aspect='auto', interpolation='nearest', vmin=vmin_special, vmax=vmax_special)
            
            ax_hm.set_xticks(range(len(pivot_full.columns)))
            ax_hm.set_xticklabels([f'{"ReLU" if x[0] else "ID"}\n{"Pruned" if x[1] > 0 else "Full"}' 
                                   for x in pivot_full.columns], fontweight='bold')
            ax_hm.set_yticks(range(len(pivot_full.index)))
            ax_hm.set_yticklabels([f'Grid {x[0]}\n{x[1]}x Width' for x in pivot_full.index], fontweight='bold')
            
            mean_pivot_full_val = np.nanmean(pivot_full.values) if np.sum(~np.isnan(pivot_full.values)) > 0 else (vmin_special + vmax_special) / 2

            for i in range(len(pivot_full.index)):
                for j in range(len(pivot_full.columns)):
                    value = pivot_full.values[i, j]
                    if not np.isnan(value):
                        text_color = 'black' if value > mean_pivot_full_val else 'white'
                        ax_hm.text(j, i, f'{value:.2f}%', ha='center', va='center', 
                                   fontweight='bold', color=text_color, fontsize=11)
            
            cbar_hm = plt.colorbar(im, ax=ax_hm, shrink=0.8)
            cbar_hm.set_label('Validation Accuracy (%)', fontsize=14, fontweight='bold')

    ax_hm.set_title('FastKAN Performance Heatmap (Special)\nValidation Accuracy Across All Configurations', 
                     fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path_png_hm = f'{output_filepath_prefix}_heatmap_special.png'
    save_path_pdf_hm = f'{output_filepath_prefix}_heatmap_special.pdf'
    plt.savefig(save_path_png_hm, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf_hm, bbox_inches='tight')
    print(f"Saved: {save_path_png_hm}")
    print(f"Saved: {save_path_pdf_hm}")
    plt.close()

# --- Main execution ---
if __name__ == "__main__":
    output_directory = "mnist_ablate_results"
    os.makedirs(output_directory, exist_ok=True)

    csv_file_path = 'mnist_ablate_results/results.csv' 
    df_for_plotting = None 

    if not os.path.exists(csv_file_path):
        print(f"Warning: CSV file not found at '{os.path.abspath(csv_file_path)}'.")
        print("Creating a dummy 'results.csv' for demonstration purposes.")
        data = {
            'grid_size': np.random.choice([4, 8, 16], 20),
            'width_mult': np.random.choice([1, 2, 4], 20),
            'use_relu': np.random.choice([True, False], 20),
            'prune_amt': np.random.choice([0, 0.1, 0.25], 20),
            'val_loss': np.random.rand(20) * 0.05, 
            'val_acc': np.random.uniform(0.98, 0.999, 20), 
            'params': np.random.randint(10000, 100000, 20),
            'flops': np.random.randint(1000000, 10000000, 20),
            'latency': np.random.rand(20) * 0.01 
        }
        dummy_df = pd.DataFrame(data)
        try:
            dummy_df.to_csv(csv_file_path, index=False)
            print(f"Dummy 'results.csv' created at '{os.path.abspath(csv_file_path)}'.")
            print("Please replace it with your actual data for meaningful results or run the script again.")
        except IOError as e:
            print(f"Error: Could not write dummy CSV file to '{os.path.abspath(csv_file_path)}': {e}")
            print("Please check directory permissions. Exiting.")
            exit()
    
    try:
        df_for_plotting = load_and_process_data(csv_file_path)
        print(f"Successfully loaded and processed data from '{os.path.abspath(csv_file_path)}'.")
    except FileNotFoundError:
        print(f"Critical Error: CSV file '{os.path.abspath(csv_file_path)}' not found.")
        print("Exiting.")
        exit()
    except ValueError as ve: 
        print(f"Error processing CSV data: {ve}")
        print("Exiting.")
        exit()
    except Exception as e: 
        print(f"An unexpected error occurred while loading or processing '{os.path.abspath(csv_file_path)}': {e}")
        print("Please check the CSV file's format and content.")
        print("Exiting.")
        exit()

    if df_for_plotting.empty:
        print(f"Warning: The DataFrame loaded from '{os.path.abspath(csv_file_path)}' is empty. "
              "Plots might be empty or incorrect. Ensure the CSV contains data.")
    
    base_filename = "fastkan_ablation"
    output_filepath_prefix_main = os.path.join(output_directory, base_filename)

    print(f"\n--- Generating Comprehensive Plot ---")
    create_stunning_plots(df_for_plotting, output_filepath_prefix_main)
    
    print(f"\n--- Generating Individual Component Figures ---")
    save_individual_component_figures(df_for_plotting, output_directory, base_filename)
    
    print(f"\n--- Generating Special Individual Plots ---")
    create_special_individual_plots(df_for_plotting, output_filepath_prefix_main)
    
    print("\n🎉 All plots created successfully!")
    print(f"📁 Files saved in: {os.path.abspath(output_directory)}")

