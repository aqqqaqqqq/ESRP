import os
import re
import numpy as np
from scipy import stats

def parse_result_line(line):
    """
    Parse a single result line and extract metrics
    
    Example line:
    scene_name: success: [True], init_potential: [tensor(0.6847)], finish_potential: [tensor(0.0822)], all_objs: [1], arrival_num: {1}
    """
    data = {}
    
    # Extract scene name
    scene_match = re.match(r'^(.+?):', line)
    if scene_match:
        data['scene_name'] = scene_match.group(1)
    
    # Extract success
    success_match = re.search(r'success:\s*\[(\w+)\]', line)
    if success_match:
        data['success'] = success_match.group(1) == 'True'
    
    # Extract init_potential
    init_pot_match = re.search(r'init_potential:\s*\[(?:tensor\()?([0-9.]+)\)?(?:, )?(?:device=\'[\w:]+\')?\]', line)
    if init_pot_match:
        data['init_potential'] = float(init_pot_match.group(1))
    
    # Extract finish_potential
    finish_pot_match = re.search(r'finish_potential:\s*\[(?:tensor\()?([0-9.]+)\)?(?:, )?(?:device=\'[\w:]+\')?\]', line)
    if finish_pot_match:
        data['fini_potential'] = float(finish_pot_match.group(1))
    
    # Extract all_objs (obj_num)
    objs_match = re.search(r'all_objs:\s*\[(\d+)\]', line)
    if objs_match:
        data['obj_num'] = int(objs_match.group(1))
    
    # Extract arrival_num
    arrival_match = re.search(r'arrival_num:\s*\{(\d+)\}', line)
    if arrival_match:
        data['arrival_num'] = int(arrival_match.group(1))
    
    return data

def analyze_results_from_folder(result_dir='tamp_results', test_data_path='test_data.txt'):
    """
    Analyze results from tamp_results folder
    
    Returns statistics grouped by object count (same logic as analyse_data.py)
    """
    all_num = [0, 0, 0]  # [1 obj, 2-3 objs, 4-6 objs]
    success_num = [0, 0, 0]
    each_arrival = [[], [], []]
    each_potential = [[], [], []]
    
    # Load test data scenes if available
    test_scenes = set()
    if os.path.exists(test_data_path):
        with open(test_data_path, 'r') as f:
            for line in f:
                scene = line.strip()
                if scene:
                    test_scenes.add(scene)
    
    # Read all result files
    if not os.path.exists(result_dir):
        print(f"Error: Directory '{result_dir}' not found!")
        return None
    
    result_files = [f for f in os.listdir(result_dir) if f.endswith('.result')]
    
    if not result_files:
        print(f"Warning: No .result files found in '{result_dir}'")
        return None
    
    print(f"Analyzing {len(result_files)} result files...")
    
    inconsistent_count = 0
    
    for filename in sorted(result_files):
        filepath = os.path.join(result_dir, filename)
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = parse_result_line(line)
                
                # Skip if parsing failed
                if 'obj_num' not in data:
                    continue
                
                # If test_data.txt exists, filter by scenes (optional)
                # For tamp_results, we typically want all results
                # if test_scenes and data.get('scene_name') not in test_scenes:
                #     continue
                
                # Categorize by object count (same logic as analyse_data.py)
                if data['obj_num'] < 2:
                    category = 0  # 1 object
                elif data['obj_num'] > 3:
                    category = 2  # 4-6 objects
                else:
                    category = 1  # 2-3 objects
                
                # Update counts
                all_num[category] += 1
                
                # Re-judge success: only count as success if ALL objects arrived
                # This is more strict than the original environment judgment
                original_success = data.get('success', False)
                actual_success = (data.get('arrival_num', 0) == data['obj_num'])
                
                if original_success != actual_success:
                    inconsistent_count += 1
                
                if actual_success:
                    success_num[category] += 1
                
                # Calculate metrics (same as analyse_data.py)
                if 'arrival_num' in data and 'obj_num' in data and data['obj_num'] > 0:
                    each_arrival[category].append(data['arrival_num'] / data['obj_num'])
                
                if 'fini_potential' in data and 'init_potential' in data and data['init_potential'] > 0:
                    each_potential[category].append(data['fini_potential'] / data['init_potential'])
    
    if inconsistent_count > 0:
        print(f"Note: Found {inconsistent_count} scenes with inconsistent success judgment (corrected)")
    
    return success_num, all_num, each_arrival, each_potential

def print_statistics(success_num, all_num, each_arrival, each_potential):
    """
    Print statistics in the same format as analyse_data.py
    """
    print("\n" + "="*70)
    print("TAMP Results Analysis")
    print("="*70)
    
    categories = ["1 object", "2-3 objects", "4-6 objects"]
    
    for i, category in enumerate(categories):
        if all_num[i] > 0:
            sr = success_num[i] / all_num[i]
            osr = sum(each_arrival[i]) / all_num[i] if each_arrival[i] else 0
            rdr = sum(each_potential[i]) / all_num[i] if each_potential[i] else 0
            
            print(f"{category:15} SR: {sr:.4f}  OSR: {osr:.4f}  RDR: {rdr:.4f}  (n={all_num[i]})")
        else:
            print(f"{category:15} No data")
    
    # Overall statistics
    success_sum = sum(success_num)
    all_sum = sum(all_num)
    
    if all_sum > 0:
        arrival_sum = sum(sum(arr) for arr in each_arrival)
        potential_sum = sum(sum(pot) for pot in each_potential)
        
        sr_all = success_sum / all_sum
        osr_all = arrival_sum / all_sum
        rdr_all = potential_sum / all_sum
        
        print("-" * 70)
        print(f"{'Overall':15} SR: {sr_all:.4f}  OSR: {osr_all:.4f}  RDR: {rdr_all:.4f}  (n={all_sum})")
    
    print("="*70)
    print("\nLegend:")
    print("  SR  = Success Rate (% of scenes fully completed)")
    print("  OSR = Object Success Rate (avg % of objects placed correctly)")
    print("  RDR = Relative Distance Reduction (final/initial potential)")
    print("="*70)

def calculate_ci_95(values):
    """
    Calculate 95% confidence interval for a list of values
    
    Args:
        values: list of numeric values
    
    Returns:
        (mean, ci_lower, ci_upper) or (mean, mean, mean) if insufficient data
    """
    if len(values) == 0:
        return 0, 0, 0
    
    mean = np.mean(values)
    
    if len(values) < 2:
        return mean, mean, mean
    
    # Calculate standard error
    std_err = stats.sem(values)
    
    # Get t-value for 95% CI with df = len(values) - 1
    t_value = stats.t.ppf(0.975, len(values) - 1)
    
    # Calculate CI
    ci = t_value * std_err
    
    return mean, mean - ci, mean + ci

def analyze_multiple_runs(result_dirs, test_data_path='test_data.txt'):
    """
    Analyze multiple runs and calculate statistics with 95% CI
    
    Args:
        result_dirs: list of result directories
        test_data_path: path to test data file
    
    Returns:
        Dictionary with statistics for each category and overall
    """
    all_runs_data = []
    
    print("="*70)
    print("Multi-Run TAMP Analysis")
    print("="*70)
    
    # Analyze each run
    for i, result_dir in enumerate(result_dirs):
        print(f"\n{'='*70}")
        print(f"Run {i+1}: {result_dir}")
        print(f"{'='*70}")
        
        results = analyze_results_from_folder(result_dir, test_data_path)
        
        if results is None:
            print(f"Warning: Failed to analyze {result_dir}")
            continue
        
        success_num, all_num, each_arrival, each_potential = results
        
        # Calculate metrics for this run
        run_metrics = {
            'run_name': result_dir,
            'categories': []
        }
        
        categories = ["1 object", "2-3 objects", "4-6 objects", "Overall"]
        
        for cat_idx in range(4):  # 3 categories + overall
            if cat_idx < 3:
                if all_num[cat_idx] > 0:
                    sr = success_num[cat_idx] / all_num[cat_idx]
                    osr = sum(each_arrival[cat_idx]) / all_num[cat_idx] if each_arrival[cat_idx] else 0
                    rdr = sum(each_potential[cat_idx]) / all_num[cat_idx] if each_potential[cat_idx] else 0
                    n = all_num[cat_idx]
                else:
                    sr = osr = rdr = 0
                    n = 0
            else:  # Overall
                total_n = sum(all_num)
                if total_n > 0:
                    sr = sum(success_num) / total_n
                    osr = sum(sum(arr) for arr in each_arrival) / total_n
                    rdr = sum(sum(pot) for pot in each_potential) / total_n
                    n = total_n
                else:
                    sr = osr = rdr = 0
                    n = 0
            
            run_metrics['categories'].append({
                'name': categories[cat_idx],
                'sr': sr,
                'osr': osr,
                'rdr': rdr,
                'n': n
            })
            
            print(f"{categories[cat_idx]:15} SR: {sr:.4f}  OSR: {osr:.4f}  RDR: {rdr:.4f}  (n={n})")
        
        all_runs_data.append(run_metrics)
    
    if not all_runs_data:
        print("\nError: No valid runs to analyze!")
        return None
    
    # Calculate aggregate statistics
    print("\n" + "="*70)
    print(f"Aggregate Statistics ({len(all_runs_data)} runs)")
    print("="*70)
    
    categories = ["1 object", "2-3 objects", "4-6 objects", "Overall"]
    
    for cat_idx, cat_name in enumerate(categories):
        # Collect metrics across all runs for this category
        sr_values = [run['categories'][cat_idx]['sr'] for run in all_runs_data]
        osr_values = [run['categories'][cat_idx]['osr'] for run in all_runs_data]
        rdr_values = [run['categories'][cat_idx]['rdr'] for run in all_runs_data]
        n_values = [run['categories'][cat_idx]['n'] for run in all_runs_data]
        
        # Calculate mean and 95% CI
        sr_mean, sr_lower, sr_upper = calculate_ci_95(sr_values)
        osr_mean, osr_lower, osr_upper = calculate_ci_95(osr_values)
        rdr_mean, rdr_lower, rdr_upper = calculate_ci_95(rdr_values)
        n_mean = np.mean(n_values)
        
        # Calculate CI margin (half-width)
        sr_ci = (sr_upper - sr_lower) / 2
        osr_ci = (osr_upper - osr_lower) / 2
        rdr_ci = (rdr_upper - rdr_lower) / 2
        
        print(f"\n{cat_name}:")
        print(f"  SR:  {sr_mean:.4f} ± {sr_ci:.4f}")
        print(f"  OSR: {osr_mean:.4f} ± {osr_ci:.4f}")
        print(f"  RDR: {rdr_mean:.4f} ± {rdr_ci:.4f}")
        print(f"  n:   {n_mean:.1f} (avg)")
    
    print("\n" + "="*70)
    print("Legend:")
    print("  SR  = Success Rate (% of scenes fully completed)")
    print("  OSR = Object Success Rate (avg % of objects placed correctly)")
    print("  RDR = Relative Distance Reduction (final/initial potential)")
    print("  ±   = 95% Confidence Interval (margin of error)")
    print("="*70)
    
    return all_runs_data

def main():
    """Main function"""
    # Define result directories to analyze
    result_dirs = [
        'tamp_results_00',
        'tamp_results_01',
        'tamp_results_02'
    ]
    
    # Check which directories exist
    existing_dirs = [d for d in result_dirs if os.path.exists(d)]
    
    if not existing_dirs:
        print("Error: None of the result directories found!")
        print("Looking for:", result_dirs)
        print("\nFalling back to single directory analysis...")
        
        # Fallback to single directory
        result_dir = 'tamp_results'
        test_data_path = 'test_data.txt'
        
        results = analyze_results_from_folder(result_dir, test_data_path)
        
        if results is None:
            print("Analysis failed!")
            return
        
        success_num, all_num, each_arrival, each_potential = results
        
        print(f"\nTotal scenes processed: {sum(all_num)}")
        print(f"Distribution: {all_num[0]} (1 obj), {all_num[1]} (2-3 objs), {all_num[2]} (4-6 objs)")
        
        print_statistics(success_num, all_num, each_arrival, each_potential)
    else:
        print(f"Found {len(existing_dirs)} result directories:")
        for d in existing_dirs:
            print(f"  - {d}")
        
        # Analyze multiple runs
        analyze_multiple_runs(existing_dirs)

if __name__ == '__main__':
    main()