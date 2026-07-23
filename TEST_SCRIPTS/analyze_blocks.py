import re
import math
import sys

def pearson_corr(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x)**2 for xi in x)
    den_y = sum((yi - mean_y)**2 for yi in y)
    
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / math.sqrt(den_x * den_y)

def spearman_corr(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    
    # Get ranks
    x_ranks = [sorted(x).index(v) for v in x]
    y_ranks = [sorted(y).index(v) for v in y]
    
    return pearson_corr(x_ranks, y_ranks)

def parse_log(filepath):
    data = []
    current_block = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("---------------------------------------"):
                if 'MSE' in current_block and 'Max_Diff_H' in current_block:
                    data.append(current_block)
                current_block = {}
                continue
                
            match_mse = re.match(r"MSE \(Original vs Inverse\):\s*([0-9\.e+-]+)", line)
            if match_mse:
                current_block['MSE'] = float(match_mse.group(1))

            match_ssi = re.match(r"SSI RhoS:\s*([0-9\.e+-]+)\s*RhoT:\s*([0-9\.e+-]+)\s*RhoU:\s*([0-9\.e+-]+)\s*RhoV:\s*([0-9\.e+-]+)", line)
            if match_ssi:
                current_block['RhoS'] = float(match_ssi.group(1))
                current_block['RhoT'] = float(match_ssi.group(2))
                current_block['RhoU'] = float(match_ssi.group(3))
                current_block['RhoV'] = float(match_ssi.group(4))
                
            match_angle = re.match(r"SSI AngleV:\s*([0-9\.e+-]+)\s*AngleH:\s*([0-9\.e+-]+)", line)
            if match_angle:
                current_block['AngleV'] = float(match_angle.group(1))
                current_block['AngleH'] = float(match_angle.group(2))
                
            match_cov_h = re.match(r"CovMatH Off-Diagonal Max \(Identity Proximity\):\s*([0-9\.e+-]+)", line)
            if match_cov_h:
                current_block['CovMatH_Identity_Prox'] = float(match_cov_h.group(1))

            match_cov_v = re.match(r"CovMatV Off-Diagonal Max \(Identity Proximity\):\s*([0-9\.e+-]+)", line)
            if match_cov_v:
                current_block['CovMatV_Identity_Prox'] = float(match_cov_v.group(1))
                
            match_eig_h = re.match(r"EigenValuesH Min Adjacent Distance:\s*([0-9\.e+-]+)", line)
            if match_eig_h:
                current_block['EigenH_Min_Adj_Dist'] = float(match_eig_h.group(1))

            match_eig_v = re.match(r"EigenValuesV Min Adjacent Distance:\s*([0-9\.e+-]+)", line)
            if match_eig_v:
                current_block['EigenV_Min_Adj_Dist'] = float(match_eig_v.group(1))
                
            match_diff = re.match(r"Forward vs Inverse Matrix Max Diff: H=([0-9\.e+-]+),\s*V=([0-9\.e+-]+)", line)
            if match_diff:
                current_block['Max_Diff_H'] = float(match_diff.group(1))
                current_block['Max_Diff_V'] = float(match_diff.group(2))

    return data

def main():
    filepath = "results/Set2/set2_0.1_testPolarityFix.comp_encoder_matrices_Y.txt"
    data = parse_log(filepath)
    
    if not data:
        print("No valid blocks found in the log.")
        return
        
    print(f"Parsed {len(data)} blocks.")
    
    targets = ['Max_Diff_H', 'Max_Diff_V', 'MSE']
    features = ['RhoS', 'RhoT', 'RhoU', 'RhoV', 'AngleV', 'AngleH', 
                'CovMatH_Identity_Prox', 'CovMatV_Identity_Prox',
                'EigenH_Min_Adj_Dist', 'EigenV_Min_Adj_Dist']
    
    print("\n--- Pearson Correlations (Linear) ---")
    for target in targets:
        print(f"\nTarget: {target}")
        target_vals = [d.get(target, 0) for d in data]
        
        corrs = {}
        for feature in features:
            feature_vals = [d.get(feature, 0) for d in data]
            corrs[feature] = pearson_corr(feature_vals, target_vals)
            
        for feature, corr in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature:<25}: {corr:.4f}")

    print("\n--- Spearman Rank Correlations (Monotonic) ---")
    for target in targets:
        print(f"\nTarget: {target}")
        target_vals = [d.get(target, 0) for d in data]
        
        corrs = {}
        for feature in features:
            feature_vals = [d.get(feature, 0) for d in data]
            corrs[feature] = spearman_corr(feature_vals, target_vals)
            
        for feature, corr in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature:<25}: {corr:.4f}")

    # Calculate average feature values for the worst blocks vs best blocks based on Max_Diff_H
    data.sort(key=lambda x: x.get('Max_Diff_H', 0), reverse=True)
    n_worst = max(1, len(data) // 20)  # top 5%
    worst_blocks = data[:n_worst]
    best_blocks = data[n_worst:]
    
    print(f"\n--- Mean feature values for Worst {n_worst} blocks vs Best {len(best_blocks)} blocks (based on Max_Diff_H) ---")
    for feature in features:
        worst_mean = sum(d.get(feature, 0) for d in worst_blocks) / len(worst_blocks)
        best_mean = sum(d.get(feature, 0) for d in best_blocks) / len(best_blocks) if len(best_blocks) > 0 else 0
        ratio = worst_mean / best_mean if best_mean != 0 else float('inf')
        print(f"  {feature:<25}: Worst={worst_mean:.6e}, Best={best_mean:.6e}, Ratio={ratio:.2f}")
        
    print(f"\n--- Mean feature values for Worst {n_worst} blocks vs Best {len(best_blocks)} blocks (based on MSE) ---")
    data.sort(key=lambda x: x.get('MSE', 0), reverse=True)
    worst_blocks = data[:n_worst]
    best_blocks = data[n_worst:]
    for feature in features:
        worst_mean = sum(d.get(feature, 0) for d in worst_blocks) / len(worst_blocks)
        best_mean = sum(d.get(feature, 0) for d in best_blocks) / len(best_blocks) if len(best_blocks) > 0 else 0
        ratio = worst_mean / best_mean if best_mean != 0 else float('inf')
        print(f"  {feature:<25}: Worst={worst_mean:.6e}, Best={best_mean:.6e}, Ratio={ratio:.2f}")

if __name__ == "__main__":
    main()
