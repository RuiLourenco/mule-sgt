import torch
import sys
import os

def check_matrices(enc_file, dec_file):
    print(f"Checking {enc_file} vs {dec_file}")
    
    if not os.path.exists(enc_file) or not os.path.exists(dec_file):
        print("One or both files do not exist.")
        return

    try:
        enc_data = torch.load(enc_file, weights_only=True)
        dec_data = torch.load(dec_file, weights_only=True)
        
        enc = enc_data[0] if isinstance(enc_data, list) else enc_data
        dec = dec_data[0] if isinstance(dec_data, list) else dec_data

        if not isinstance(enc, torch.Tensor) or not isinstance(dec, torch.Tensor):
            print("Loaded data is not a tensor.")
            return

        print(f"Encoder tensor size: {enc.size()}")
        print(f"Decoder tensor size: {dec.size()}")

        diff = torch.abs(enc - dec).max().item()
        print(f"Max absolute difference (Encoder vs Decoder): {diff:.10e}")
        if diff == 0:
            print("✅ Matrices are identically matched!")
        else:
            print("❌ Matrices have differences!")

        # Stability check: T @ T^T should be Identity (lossless)
        ident_enc = enc @ enc.T
        ident_true = torch.eye(enc.size(0), dtype=enc.dtype, device=enc.device)
        enc_stability = torch.abs(ident_enc - ident_true).max().item()
        print(f"Encoder stability max diff from Identity: {enc_stability:.10e}")

        ident_dec = dec @ dec.T
        dec_stability = torch.abs(ident_dec - ident_true).max().item()
        print(f"Decoder stability max diff from Identity: {dec_stability:.10e}")
        
        if enc_stability < 1e-5 and dec_stability < 1e-5:
            print("✅ Matrices are stable and orthogonal.")
        else:
            print("❌ Matrices might not be perfectly orthogonal (or stable).")

    except Exception as e:
        print(f"Error checking matrices: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 check_matrices.py <encoder_file.pt> <decoder_file.pt>")
    else:
        check_matrices(sys.argv[1], sys.argv[2])