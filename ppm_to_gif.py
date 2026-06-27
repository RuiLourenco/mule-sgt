import os
import re
import argparse
import subprocess
import tempfile

def generate_zigzag_webm(input_dir, output_file, fps):
    # Regex to match NNN_MMM.ppm or NNN_MMM.png
    pattern = re.compile(r'^(\d+)_(\d+)\.(?:ppm|png)$', re.IGNORECASE)
    
    parsed_files = []
    
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    # 1. Grab and parse files
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            filepath = os.path.abspath(os.path.join(input_dir, filename))
            parsed_files.append((row, col, filepath))
            
    if not parsed_files:
        print(f"No files matching 'NNN_MMM.ppm' or 'NNN_MMM.png' found.")
        return

    # 2. Sort Logic (Horizontal-first Zigzag)
    rows = {}
    for row, col, filepath in parsed_files:
        if row not in rows:
            rows[row] = []
        rows[row].append((col, filepath))

    sorted_row_keys = sorted(rows.keys())
    ordered_filepaths = []
    
    for i, row_key in enumerate(sorted_row_keys):
        cols = rows[row_key]
        # Even rows: Left-to-Right | Odd rows: Right-to-Left
        is_reverse = (i % 2 != 0)
        cols.sort(key=lambda x: x[0], reverse=is_reverse)
        for col, filepath in cols:
            ordered_filepaths.append(filepath)

    print(f"Ordered {len(ordered_filepaths)} frames for zigzag scan.")

    # 3. Create a temporary 'concat' file for FFmpeg
    # This prevents 'argument list too long' errors on large datasets
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for path in ordered_filepaths:
            # FFmpeg concat format requires escaped paths or quotes
            f.write(f"file '{path}'\n")
            # Set duration for each frame (1/fps)
            f.write(f"duration {1/fps}\n")
        
        # FFmpeg quirk: the last file should be repeated or have a duration
        f.write(f"file '{ordered_filepaths[-1]}'\n")
        concat_file = f.name

    # 4. Run FFmpeg
    # -f concat: use the text file list
    # -i: input file
    # -c:v libvpx-vp9: use VP9 codec
    # -lossless 1: EXACTLY what you asked for (mathematically lossless)
    # -pix_fmt yuv420p: ensures compatibility with most players
    
    command = [
        'ffmpeg', '-y', 
        '-f', 'concat', '-safe', '0', 
        '-i', concat_file,
        '-c:v', 'libvpx-vp9', 
        '-lossless', '1', 
        '-pix_fmt', 'yuv420p',
        output_file
    ]

    try:
        print("Encoding WebM (Lossless VP9)...")
        subprocess.run(command, check=True)
        print(f"\nSuccess! File saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    finally:
        if os.path.exists(concat_file):
            os.remove(concat_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPM/PNG to Lossless WebM Zigzag Converter")
    parser.add_argument("-i", "--input", required=True, help="Input directory")
    parser.add_argument("-o", "--output", required=True, help="Output file (e.g. video.webm)")
    parser.add_argument("-s", "--speed", type=float, default=10.0, help="Frames Per Second")
    
    args = parser.parse_args()
    generate_zigzag_webm(args.input, args.output, args.speed)