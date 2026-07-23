import re
import sys

def parse_log(filepath):
    data = []
    current_block = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("---------------------------------------"):
                if "MSE" in current_block and "Max_Diff_H" in current_block:
                    data.append(current_block)
                current_block = {}
                continue
            
            match_pos = re.match(r"Block Position:\s*([0-9\-]+)\s+([0-9\-]+)\s+([0-9\-]+)\s+([0-9\-]+)", line)
            if match_pos:
                current_block["Pos"] = f"{match_pos.group(1)}_{match_pos.group(2)}_{match_pos.group(3)}_{match_pos.group(4)}"
                
            match_mse = re.match(r"MSE \(Original vs Inverse\):\s*([0-9\.e+-]+)", line)
            if match_mse: current_block["MSE"] = float(match_mse.group(1))

            match_eig_h = re.match(r"EigenValuesH Min Adjacent Distance:\s*([0-9\.e+-]+)", line)
            if match_eig_h: current_block["EigenH_Min_Adj_Dist"] = float(match_eig_h.group(1))

            match_eig_v = re.match(r"EigenValuesV Min Adjacent Distance:\s*([0-9\.e+-]+)", line)
            if match_eig_v: current_block["EigenV_Min_Adj_Dist"] = float(match_eig_v.group(1))
                
            match_diff = re.match(r"Forward vs Inverse Matrix Max Diff: H=([0-9\.e+-]+),\s*V=([0-9\.e+-]+)", line)
            if match_diff:
                current_block["Max_Diff_H"] = float(match_diff.group(1))
                current_block["Max_Diff_V"] = float(match_diff.group(2))
    return data

def print_stats(data, name):
    print(f"\n--- {name} ---")
    if not data:
        print("No blocks found.")
        return
    max_mse = max([d.get("MSE", 0) for d in data])
    max_diff_h = max([d.get("Max_Diff_H", 0) for d in data])
    max_diff_v = max([d.get("Max_Diff_V", 0) for d in data])
    min_eig = min([min(d.get("EigenH_Min_Adj_Dist", 1), d.get("EigenV_Min_Adj_Dist", 1)) for d in data])
    blocks_below_1e14 = sum(1 for d in data if min(d.get("EigenH_Min_Adj_Dist", 1), d.get("EigenV_Min_Adj_Dist", 1)) < 1e-14)
    
    print(f"Total blocks:   {len(data)}")
    print(f"Max MSE:        {max_mse:.6e}")
    print(f"Max Diff H:     {max_diff_h:.6e}")
    print(f"Max Diff V:     {max_diff_v:.6e}")
    print(f"Min Eigen Dist: {min_eig:.6e}")
    print(f"Blocks < 1e-14: {blocks_below_1e14}")

if __name__ == "__main__":
    d1 = parse_log("results/Set2/set2_0.1_testPolarityFix.comp_encoder_matrices_Y.txt")
    d2 = parse_log("results/Set2/set2_0.1_testMinEigProx.comp_encoder_matrices_Y.txt")
    print_stats(d1, "OLD (Without Fix - testPolarityFix)")
    print_stats(d2, "NEW (With Fix - testMinEigProx)")
