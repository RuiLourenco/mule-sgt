import matplotlib.pyplot as plt
import pandas as pd

def plot_rd_graph(csv_file, output_file, reference_data=None):
    """
    Plots a PSNR-YUV/Rate RD-Graph from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the data.
        output_file (str): Path to save the output graph.
        reference_data (dict, optional): Dictionary with 'x' and 'y' keys for reference plot.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Extract Rate and PSNR-YUV columns
    x = data['Rate']
    y = data['Mean_PSNR_YUV']

    # Plot the main data
    plt.semilogx(x, y, marker='o', label='Mule-SGT')

    # Plot reference data if provided
    if reference_data:
        plt.semilogx(reference_data['x'], reference_data['y'], marker='x', label='Mule-Slant', linestyle='--')

    # Add labels, title, and legend
    plt.xlabel('Rate (bpp)')
    plt.ylabel('PSNR-YUV (dB)')
    plt.title('RD Comparison')
    plt.legend()

    # Save the graph
    plt.savefig(output_file)
    print(f"Graph saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Path to the CSV file
    csv_file = 'metrics_results.csv'

    # Output file for the graph
    output_file = 'rd_curves_psnr_yuv.png'

    # Optional reference data (to be added later)
    reference_data = {
        'x': [0.0055,0.0229,0.1050,0.8408],
        'y': [33.64,38.17,41.36,46.01]

    }

    # Call the function to plot the graph
    plot_rd_graph(csv_file, output_file, reference_data=reference_data)