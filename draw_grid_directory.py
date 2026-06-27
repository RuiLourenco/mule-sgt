import os
import glob
import argparse
from PIL import Image, ImageDraw

def draw_red_grid(input_path: str, output_path: str, block_width: int, block_height: int):
    # Pillow automatically detects the format from the file extension for opening
    with Image.open(input_path) as img:
        
        # Convert to standard RGB to prevent color mode errors when saving to formats like PNG
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        # Draw vertical grid lines in red
        for x in range(0, img_width, block_width):
            draw.line([(x, 0), (x, img_height)], fill=(255, 0, 0))

        # Draw horizontal grid lines in red
        for y in range(0, img_height, block_height):
            draw.line([(0, y), (img_width, y)], fill=(255, 0, 0))

        # Pillow detects the output_path extension (.png, .ppm, etc.) and formats accordingly
        img.save(output_path)

def process_directory(input_dir: str, output_dir: str, block_width: int, block_height: int):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Gather all .ppm and .png files (case-insensitive where possible, though glob relies on OS)
    search_patterns = [
        os.path.join(input_dir, "*.ppm"),
        os.path.join(input_dir, "*.PPM"),
        os.path.join(input_dir, "*.png"),
        os.path.join(input_dir, "*.PNG")
    ]
    
    files_to_process = []
    for pattern in search_patterns:
        files_to_process.extend(glob.glob(pattern))

    # Remove duplicates in case of case-insensitive file systems (like Windows)
    files_to_process = list(set(files_to_process))

    if not files_to_process:
        print(f"No .ppm or .png files found in '{input_dir}'.")
        return

    print(f"Found {len(files_to_process)} images to process.")

    for input_path in files_to_process:
        # Extract the original filename without the extension
        base_name = os.path.basename(input_path)
        name_without_ext, _ = os.path.splitext(base_name)
        
        # Force the output extension to be .png
        output_filename = f"{name_without_ext}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        draw_red_grid(input_path, output_path, block_width, block_height)
        print(f"Saved: {output_filename}")

    print(f"Batch processing complete. Files saved to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch apply red grid lines to images.")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing images")
    parser.add_argument("-o", "--output", required=True, help="Output directory for processed images")
    parser.add_argument("-w", "--width", type=int, required=True, help="Grid block width in pixels")
    parser.add_argument("-H", "--height", type=int, required=True, help="Grid block height in pixels")

    args = parser.parse_args()
    
    process_directory(args.input, args.output, args.width, args.height)