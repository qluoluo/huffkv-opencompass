import os
import json
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def load_attention_stats(stats_file):
    """Load attention statistics from a JSON file"""
    with open(stats_file, 'r') as f:
        return json.load(f)

def aggregate_statistics(result_dir):
    """
    Aggregate attention statistics across all datasets, samples, layers, and heads
    """
    # Find all statistics files
    stats_files = glob.glob(os.path.join(result_dir, "**", "all_layers_attn_stats.json"), recursive=True)
    print(f"Found {len(stats_files)} statistics files")
    
    # Dictionary to collect all values for each metric
    # Structure: metric_name -> list of all values across all files/layers/heads
    all_metrics = defaultdict(list)
    
    # Metrics we want to track
    metric_names = ['max', 'mean', 'min', 'top1%', 'top5%', 'top10%', 'top20%', 'top50%']
    
    # Process each file
    for stats_file in tqdm(stats_files, desc="Processing files"):
        try:
            stats_data = load_attention_stats(stats_file)
            
            # Iterate through each layer
            for layer_name, layer_stats_list in stats_data.items():
                # Each layer may have multiple forward passes (if any)
                for forward_stats in layer_stats_list:
                    # Iterate through each head
                    for head_name, head_stats in forward_stats.items():
                        # Collect each metric
                        for metric_name in metric_names:
                            if metric_name in head_stats:
                                all_metrics[metric_name].append(head_stats[metric_name])
        
        except Exception as e:
            print(f"Error processing {stats_file}: {e}")
            continue
    
    # Compute aggregated statistics
    aggregated_stats = {}
    
    for metric_name in metric_names:
        if metric_name in all_metrics and len(all_metrics[metric_name]) > 0:
            values = np.array(all_metrics[metric_name])
            
            aggregated_stats[metric_name] = {
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'min': float(np.min(values)),
                'std': float(np.std(values)),
                'count': int(len(values))
            }
        else:
            aggregated_stats[metric_name] = {
                'max': None,
                'mean': None,
                'min': None,
                'std': None,
                'count': 0
            }
    
    return aggregated_stats, all_metrics

def aggregate_by_dimension(result_dir):
    """
    Aggregate statistics with breakdown by dataset, layer, and head
    """
    stats_files = glob.glob(os.path.join(result_dir, "**", "all_layers_attn_stats.json"), recursive=True)
    print(f"Found {len(stats_files)} statistics files")
    
    metric_names = ['max', 'mean', 'min', 'top1%', 'top5%', 'top10%', 'top20%', 'top50%']
    
    # Collect statistics by different dimensions
    by_dataset = defaultdict(lambda: defaultdict(list))
    by_layer = defaultdict(lambda: defaultdict(list))
    by_head = defaultdict(lambda: defaultdict(list))
    
    for stats_file in tqdm(stats_files, desc="Processing files by dimension"):
        try:
            # Extract dataset name from path
            path_parts = stats_file.split(os.sep)
            model_idx = -1
            for i, part in enumerate(path_parts):
                if 'Llama' in part or 'llama' in part or 'model' in part.lower():
                    model_idx = i
                    break
            
            if model_idx >= 0 and model_idx + 1 < len(path_parts):
                dataset_name = path_parts[model_idx + 1]
            else:
                dataset_name = "unknown"
            
            stats_data = load_attention_stats(stats_file)
            
            # Iterate through each layer
            for layer_name, layer_stats_list in stats_data.items():
                for forward_stats in layer_stats_list:
                    # Iterate through each head
                    for head_name, head_stats in forward_stats.items():
                        # Collect each metric
                        for metric_name in metric_names:
                            if metric_name in head_stats:
                                value = head_stats[metric_name]
                                
                                # By dataset
                                by_dataset[dataset_name][metric_name].append(value)
                                
                                # By layer
                                by_layer[layer_name][metric_name].append(value)
                                
                                # By head
                                by_head[head_name][metric_name].append(value)
        
        except Exception as e:
            print(f"Error processing {stats_file}: {e}")
            continue
    
    # Compute aggregated statistics for each dimension
    def compute_stats(data_dict):
        result = {}
        for key, metrics in data_dict.items():
            result[key] = {}
            for metric_name in metric_names:
                if metric_name in metrics and len(metrics[metric_name]) > 0:
                    values = np.array(metrics[metric_name])
                    result[key][metric_name] = {
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'min': float(np.min(values)),
                        'std': float(np.std(values)),
                        'count': int(len(values))
                    }
        return result
    
    return {
        'by_dataset': compute_stats(by_dataset),
        'by_layer': compute_stats(by_layer),
        'by_head': compute_stats(by_head)
    }

def save_aggregated_results(result_dir, output_dir=None):
    """
    Main function to compute and save aggregated statistics
    """
    if output_dir is None:
        output_dir = result_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*50)
    print("Computing overall aggregated statistics...")
    print("="*50)
    
    # Overall aggregation
    aggregated_stats, all_metrics = aggregate_statistics(result_dir)
    
    # Save overall statistics
    output_file = os.path.join(output_dir, "aggregated_attention_stats.json")
    with open(output_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    
    print(f"\nOverall statistics saved to: {output_file}")
    print("\nOverall Statistics Summary:")
    print(json.dumps(aggregated_stats, indent=2))
    
    print("\n" + "="*50)
    print("Computing dimension-wise aggregated statistics...")
    print("="*50)
    
    # Dimension-wise aggregation
    dimensional_stats = aggregate_by_dimension(result_dir)
    
    # Save dimensional statistics
    output_file_dim = os.path.join(output_dir, "aggregated_attention_stats_by_dimension.json")
    with open(output_file_dim, 'w') as f:
        json.dump(dimensional_stats, f, indent=2)
    
    print(f"\nDimensional statistics saved to: {output_file_dim}")
    
    # Print summary
    print("\n" + "="*50)
    print("Summary by Dataset:")
    print("="*50)
    for dataset, stats in dimensional_stats['by_dataset'].items():
        print(f"\n{dataset}:")
        if 'max' in stats:
            print(f"  Sample count: {stats['max'].get('count', 0)}")
    
    print("\n" + "="*50)
    print("Summary by Layer:")
    print("="*50)
    layer_names = sorted(dimensional_stats['by_layer'].keys(), 
                        key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    for layer in layer_names[:3]:  # Print first 3 layers
        stats = dimensional_stats['by_layer'][layer]
        print(f"\n{layer}:")
        if 'max' in stats:
            print(f"  Sample count: {stats['max'].get('count', 0)}")
    print("  ...")
    
    # Save raw values for further analysis (optional, can be large)
    print("\n" + "="*50)
    print("Saving raw metric values...")
    print("="*50)
    
    raw_values_file = os.path.join(output_dir, "raw_metric_values.npz")
    np.savez(raw_values_file, **{k: np.array(v) for k, v in all_metrics.items()})
    print(f"Raw values saved to: {raw_values_file}")
    
    return aggregated_stats, dimensional_stats

if __name__ == "__main__":
    # root_dir = '/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105'
    root_dir = '/inspire/hdd/project/heziweiproject/heziwei-25044'
    
    result_dir = os.path.join(
        root_dir, 
        "projects_zyning/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
    )
    
    # You can specify a different output directory if needed
    output_dir = os.path.join(result_dir, "aggregated_results")
    
    print(f"Processing results from: {result_dir}")
    print(f"Saving aggregated results to: {output_dir}")
    
    aggregated_stats, dimensional_stats = save_aggregated_results(result_dir, output_dir)
    
    print("\n" + "="*50)
    print("Aggregation complete!")
    print("="*50)
    
    # Print final summary
    print("\nFinal Overall Statistics:")
    for metric_name, stats in aggregated_stats.items():
        print(f"\n{metric_name}:")
        print(f"  max:   {stats['max']:.6f}" if stats['max'] is not None else "  max:   None")
        print(f"  mean:  {stats['mean']:.6f}" if stats['mean'] is not None else "  mean:  None")
        print(f"  min:   {stats['min']:.6f}" if stats['min'] is not None else "  min:   None")
        print(f"  count: {stats['count']}")
