import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

def load_attention_stats(stats_file):
    """Load attention statistics from a JSON file"""
    with open(stats_file, 'r') as f:
        return json.load(f)

def extract_attention_matrix(result_dir, metric='max'):
    """
    Extract attention values for all layers and heads
    Returns:
        samples_data: list of dicts, each dict contains layer->head->value mapping
        layer_names: sorted list of layer names
        head_names: sorted list of head names
    """
    stats_files = glob.glob(os.path.join(result_dir, "**", "all_layers_attn_stats.json"), recursive=True)
    print(f"Found {len(stats_files)} statistics files")
    
    samples_data = []
    all_layers = set()
    all_heads = set()
    
    for stats_file in tqdm(stats_files, desc="Loading attention data"):
        try:
            stats_data = load_attention_stats(stats_file)
            
            sample_dict = defaultdict(dict)
            
            # Iterate through each layer
            for layer_name, layer_stats_list in stats_data.items():
                all_layers.add(layer_name)
                
                # Take the first forward pass (or average if multiple)
                if len(layer_stats_list) > 0:
                    forward_stats = layer_stats_list[0]
                    
                    # Iterate through each head
                    for head_name, head_stats in forward_stats.items():
                        all_heads.add(head_name)
                        
                        if metric in head_stats:
                            sample_dict[layer_name][head_name] = head_stats[metric]
            
            if sample_dict:
                samples_data.append(sample_dict)
        
        except Exception as e:
            print(f"Error processing {stats_file}: {e}")
            continue
    
    # Sort layer and head names
    layer_names = sorted(all_layers, key=lambda x: int(x.split('_')[1]) if '_' in x and len(x.split('_')) > 1 else 0)
    head_names = sorted(all_heads, key=lambda x: int(x.split('_')[1]) if '_' in x and len(x.split('_')) > 1 else 0)
    
    return samples_data, layer_names, head_names

def create_attention_matrix(samples_data, layer_names, head_names, sample_idx=0):
    """
    Create a 2D matrix for visualization
    Rows: Layers
    Columns: Heads
    """
    num_layers = len(layer_names)
    num_heads = len(head_names)
    
    matrix = np.zeros((num_layers, num_heads))
    
    if sample_idx < len(samples_data):
        sample_data = samples_data[sample_idx]
        
        for i, layer_name in enumerate(layer_names):
            for j, head_name in enumerate(head_names):
                if layer_name in sample_data and head_name in sample_data[layer_name]:
                    matrix[i, j] = sample_data[layer_name][head_name]
                else:
                    matrix[i, j] = np.nan
    
    return matrix

def create_average_attention_matrix(samples_data, layer_names, head_names):
    """
    Create averaged 2D matrix across all samples
    """
    num_layers = len(layer_names)
    num_heads = len(head_names)
    
    sum_matrix = np.zeros((num_layers, num_heads))
    count_matrix = np.zeros((num_layers, num_heads))
    
    for sample_data in samples_data:
        for i, layer_name in enumerate(layer_names):
            for j, head_name in enumerate(head_names):
                if layer_name in sample_data and head_name in sample_data[layer_name]:
                    sum_matrix[i, j] += sample_data[layer_name][head_name]
                    count_matrix[i, j] += 1
    
    # Avoid division by zero
    avg_matrix = np.divide(sum_matrix, count_matrix, 
                          out=np.full_like(sum_matrix, np.nan), 
                          where=count_matrix != 0)
    
    return avg_matrix

def visualize_attention_heatmap(matrix, layer_names, head_names, title, output_path, metric_name='max'):
    """
    Visualize attention values as a heatmap
    """
    fig, ax = plt.subplots(figsize=(max(12, len(head_names) * 0.3), 
                                    max(10, len(layer_names) * 0.3)))
    
    # Create heatmap
    sns.heatmap(matrix, 
                cmap='YlOrRd',  # Yellow-Orange-Red colormap
                cbar_kws={'label': f'Attention {metric_name} value'},
                ax=ax,
                xticklabels=[h.replace('head_', 'H') for h in head_names],
                yticklabels=[l.replace('layer_', 'L') for l in layer_names],
                linewidths=0.5,
                linecolor='gray',
                square=False,
                vmin=np.nanmin(matrix),
                vmax=np.nanmax(matrix))
    
    ax.set_xlabel('Head Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add statistics text
    stats_text = f'Min: {np.nanmin(matrix):.4f}\n'
    stats_text += f'Max: {np.nanmax(matrix):.4f}\n'
    stats_text += f'Mean: {np.nanmean(matrix):.4f}\n'
    stats_text += f'Std: {np.nanstd(matrix):.4f}'
    
    plt.text(1.02, 0.5, stats_text, 
             transform=ax.transAxes,
             fontsize=10,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()

def visualize_all_metrics(result_dir, output_dir=None):
    """
    Create visualizations for all metrics
    """
    if output_dir is None:
        output_dir = os.path.join(result_dir, "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['max', 'mean', 'min', 'top1%', 'top5%', 'top10%', 'top20%', 'top50%']
    
    for metric in metrics:
        print(f"\n{'='*50}")
        print(f"Processing metric: {metric}")
        print(f"{'='*50}")
        
        # Extract data
        samples_data, layer_names, head_names = extract_attention_matrix(result_dir, metric=metric)
        
        if not samples_data:
            print(f"No data found for metric {metric}")
            continue
        
        print(f"Found {len(samples_data)} samples")
        print(f"Layers: {len(layer_names)}, Heads: {len(head_names)}")
        
        # First sample visualization
        print(f"\nCreating visualization for first sample...")
        first_sample_matrix = create_attention_matrix(samples_data, layer_names, head_names, sample_idx=0)
        visualize_attention_heatmap(
            first_sample_matrix,
            layer_names,
            head_names,
            f'Attention {metric.upper()} - First Sample\n({len(layer_names)} layers × {len(head_names)} heads)',
            os.path.join(output_dir, f'attention_{metric}_first_sample.png'),
            metric_name=metric
        )
        
        # Average across samples visualization
        print(f"Creating visualization for average across samples...")
        avg_matrix = create_average_attention_matrix(samples_data, layer_names, head_names)
        visualize_attention_heatmap(
            avg_matrix,
            layer_names,
            head_names,
            f'Attention {metric.upper()} - Average Across {len(samples_data)} Samples\n({len(layer_names)} layers × {len(head_names)} heads)',
            os.path.join(output_dir, f'attention_{metric}_average.png'),
            metric_name=metric
        )
    
    print(f"\n{'='*50}")
    print("All visualizations completed!")
    print(f"{'='*50}")

def visualize_single_metric(result_dir, metric='max', output_dir=None):
    """
    Create visualizations for a single metric
    """
    if output_dir is None:
        output_dir = os.path.join(result_dir, "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Processing metric: {metric}")
    print(f"{'='*50}")
    
    # Extract data
    samples_data, layer_names, head_names = extract_attention_matrix(result_dir, metric=metric)
    
    if not samples_data:
        print(f"No data found for metric {metric}")
        return
    
    print(f"Found {len(samples_data)} samples")
    print(f"Layers: {len(layer_names)}, Heads: {len(head_names)}")
    
    # First sample visualization
    print(f"\nCreating visualization for first sample...")
    first_sample_matrix = create_attention_matrix(samples_data, layer_names, head_names, sample_idx=0)
    visualize_attention_heatmap(
        first_sample_matrix,
        layer_names,
        head_names,
        f'Attention {metric.upper()} - First Sample\n({len(layer_names)} layers × {len(head_names)} heads)',
        os.path.join(output_dir, f'attention_{metric}_first_sample.png'),
        metric_name=metric
    )
    
    # Average across samples visualization
    print(f"Creating visualization for average across samples...")
    avg_matrix = create_average_attention_matrix(samples_data, layer_names, head_names)
    visualize_attention_heatmap(
        avg_matrix,
        layer_names,
        head_names,
        f'Attention {metric.upper()} - Average Across {len(samples_data)} Samples\n({len(layer_names)} layers × {len(head_names)} heads)',
        os.path.join(output_dir, f'attention_{metric}_average.png'),
        metric_name=metric
    )
    
    print(f"\n{'='*50}")
    print("Visualization completed!")
    print(f"{'='*50}")
    
    return first_sample_matrix, avg_matrix, layer_names, head_names

if __name__ == "__main__":
    # root_dir = '/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105'
    root_dir = '/inspire/hdd/project/heziweiproject/heziwei-25044'
    
    result_dir = os.path.join(
        root_dir, 
        "projects_zyning/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
    )
    
    output_dir = os.path.join(result_dir, "visualizations")
    
    print(f"Processing results from: {result_dir}")
    print(f"Saving visualizations to: {output_dir}")
    
    # Option 1: Visualize only 'max' metric
    visualize_single_metric(result_dir, metric='max', output_dir=output_dir)
    
    # Option 2: Visualize all metrics (uncomment to use)
    # visualize_all_metrics(result_dir, output_dir=output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")
