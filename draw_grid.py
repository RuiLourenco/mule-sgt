import sys
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
        print(f"Success: Image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python draw_grid.py <input_image> <output_image> <block_width> <block_height>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])

    draw_red_grid(input_file, output_file, width, height)