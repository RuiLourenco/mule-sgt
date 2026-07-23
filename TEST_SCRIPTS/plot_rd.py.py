import argparse
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_rate_distortion(csv_files: list[Path], output_filename: str):
    """
    Reads a list of CSV files containing 'rate' and 'psnr' columns,
    and plots a Rate-Distortion curve with a logarithmic x-axis.
    """
    plt.figure(figsize=(8, 6))

    # Replicating the original marker styles: Red Square, Green Circle, Blue Up-Triangle, Black Down-Triangle
    styles = itertools.cycle([
        {'color': 'red', 'marker': 's'},
        {'color': 'lime', 'marker': 'o'},
        {'color': 'blue', 'marker': '^'},
        {'color': 'black', 'marker': 'v'}
    ])

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            style = next(styles)
            label = file_path.stem  # Extracts the filename without the .csv extension
            
            # Ensure the CSV contains the expected columns
            if 'rate' not in df.columns or 'psnr' not in df.columns:
                print(f"Skipping {file_path}: Missing 'rate' or 'psnr' column headers.")
                continue

            plt.plot(
                df['rate'],
                df['psnr'],
                label=label,
                color=style['color'],
                marker=style['marker'],
                linestyle='-',
                linewidth=1,
                markersize=6
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Set x-axis to logarithmic scale to match the provided image
    plt.xscale('log')

    # Configure major and minor grid lines
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3, color='thistle')

    # Set axis labels and formatting
    plt.xlabel('rate [bpp]', fontsize=14)
    plt.ylabel('PSNR-YUV [dB]', fontsize=14)
    
    # Configure the legend to sit in the upper left without a box
    plt.legend(loc='upper left', frameon=False, fontsize=12)

    # Save the output file
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Rate-Distortion curves from CSV files.")
    parser.add_argument("csv_files", nargs='+', type=Path, help="Paths to the input CSV files.")
    parser.add_argument("-o", "--output", default="rd_curve.png", help="Output PNG filename (default: rd_curve.png)")

    args = parser.parse_args()
    plot_rate_distortion(args.csv_files, args.output)