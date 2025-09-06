import os
import json
import random
from typing import Tuple, List
import math
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Text samples for different difficulty levels
EASY_TEXTS = [
    "Hello World",
    "Welcome to OCR",
    "Computer Vision",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "Data Science",
    "Python Programming",
    "Image Processing",
    "Text Recognition"
]

MEDIUM_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
    "Python is a powerful programming language",
    "Computer vision enables machines to interpret visual information",
    "Optical Character Recognition extracts text from images",
    "Machine learning algorithms learn from data patterns",
    "Deep neural networks have revolutionized AI",
    "Natural language processing handles human language",
    "Data preprocessing is crucial for ML success",
    "Feature extraction improves model performance"
]

HARD_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.",
    "Optical Character Recognition (OCR) is a technology that converts different types of documents into editable and searchable data.",
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural language processing combines computational linguistics with machine learning to help computers understand human language.",
    "Data preprocessing involves cleaning and transforming raw data into a format suitable for machine learning algorithms.",
    "Feature engineering is the process of selecting, modifying, or creating new features from raw data to improve model performance.",
    "Model evaluation is crucial for understanding how well a machine learning model performs on unseen data."
]

HANDWRITTEN_STYLES = [
    "cursive", "print", "mixed"
]

FONT_SIZES = [16, 20, 24, 28, 32, 36, 40, 48]
ROTATION_ANGLES = [0, 15, 30, 45, -15, -30, -45, 90, -90, 180]

def get_random_color(exclude=None):
    """Generate a random color excluding specified colors."""
    exclude = set(exclude or [])
    colors = [
        (0, 0, 0),      # Black
        (255, 255, 255), # White
        (220, 20, 60),   # Crimson
        (65, 105, 225),  # Royal Blue
        (60, 179, 113),  # Medium Sea Green
        (255, 165, 0),   # Orange
        (148, 0, 211),   # Dark Violet
        (0, 191, 255),   # Deep Sky Blue
        (255, 99, 71),   # Tomato
        (154, 205, 50),  # Yellow Green
        (128, 128, 128), # Gray
        (255, 20, 147),  # Deep Pink
    ]
    while True:
        color = random.choice(colors)
        if color not in exclude:
            return color

def create_text_image(text: str, width: int, height: int, font_size: int = 24, 
                     font_color: Tuple[int, int, int] = (0, 0, 0),
                     bg_color: Tuple[int, int, int] = (255, 255, 255),
                     rotation: int = 0, noise_level: float = 0.0) -> Image.Image:
    """Create an image with text."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Wrap text to fit image width
    wrapped_text = textwrap.fill(text, width=width//(font_size//2))
    lines = wrapped_text.split('\n')
    
    # Calculate text position (centered)
    line_height = font_size + 5
    total_height = len(lines) * line_height
    start_y = (height - total_height) // 2
    
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = start_y + i * line_height
        draw.text((x, y), line, font=font, fill=font_color)
    
    # Apply rotation if specified
    if rotation != 0:
        img = img.rotate(rotation, expand=True, fillcolor=bg_color)
        # Crop back to original size
        w, h = img.size
        left = (w - width) // 2
        top = (h - height) // 2
        img = img.crop((left, top, left + width, top + height))
    
    # Add noise if specified
    if noise_level > 0:
        img = add_noise(img, noise_level)
    
    return img

def add_noise(img: Image.Image, noise_level: float) -> Image.Image:
    """Add random noise to image."""
    import numpy as np
    img_array = np.array(img)
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def create_handwritten_style_image(text: str, width: int, height: int, 
                                  style: str = "print") -> Image.Image:
    """Create an image with handwritten-style text."""
    font_size = random.choice([20, 24, 28, 32])
    font_color = get_random_color()
    bg_color = get_random_color(exclude=[font_color])
    
    # Add some randomness to simulate handwriting
    rotation = random.choice([0, 2, -2, 4, -4])
    noise_level = random.uniform(0.02, 0.05)
    
    return create_text_image(text, width, height, font_size, font_color, 
                           bg_color, rotation, noise_level)

def create_document_style_image(text: str, width: int, height: int) -> Image.Image:
    """Create a document-style image with formatted text."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add some document-like elements
    # Draw lines
    for i in range(0, height, 30):
        draw.line([(50, i), (width-50, i)], fill=(200, 200, 200), width=1)
    
    # Add text
    lines = textwrap.fill(text, width=width//10).split('\n')
    y = 20
    for line in lines:
        if y + 20 < height:
            draw.text((60, y), line, font=font, fill=(0, 0, 0))
            y += 25
    
    return img

def generate_dataset(output_dir: str, num_images: int = 15, image_size: int = 512):
    """Generate OCR test dataset."""
    os.makedirs(output_dir, exist_ok=True)
    input_dir = os.path.join(output_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    
    ground_truth = {}
    
    # Generate different types of images
    for i in range(1, num_images + 1):
        img_type = random.choice(["easy", "medium", "hard", "handwritten", "document", "rotated", "noisy"])
        
        if img_type == "easy":
            text = random.choice(EASY_TEXTS)
            font_size = random.choice([24, 28, 32])
            font_color = (0, 0, 0)
            bg_color = (255, 255, 255)
            img = create_text_image(text, image_size, image_size//2, font_size, font_color, bg_color)
            
        elif img_type == "medium":
            text = random.choice(MEDIUM_TEXTS)
            font_size = random.choice([16, 20, 24])
            font_color = get_random_color()
            bg_color = get_random_color(exclude=[font_color])
            img = create_text_image(text, image_size, image_size, font_size, font_color, bg_color)
            
        elif img_type == "hard":
            text = random.choice(HARD_TEXTS)
            font_size = random.choice([14, 16, 18])
            font_color = get_random_color()
            bg_color = get_random_color(exclude=[font_color])
            noise_level = random.uniform(0.01, 0.03)
            img = create_text_image(text, image_size, image_size, font_size, font_color, bg_color, noise_level=noise_level)
            
        elif img_type == "handwritten":
            text = random.choice(MEDIUM_TEXTS)
            style = random.choice(HANDWRITTEN_STYLES)
            img = create_handwritten_style_image(text, image_size, image_size//2, style)
            
        elif img_type == "document":
            text = random.choice(HARD_TEXTS)
            img = create_document_style_image(text, image_size, image_size)
            
        elif img_type == "rotated":
            text = random.choice(MEDIUM_TEXTS)
            font_size = random.choice([20, 24, 28])
            font_color = get_random_color()
            bg_color = get_random_color(exclude=[font_color])
            rotation = random.choice(ROTATION_ANGLES)
            img = create_text_image(text, image_size, image_size//2, font_size, font_color, bg_color, rotation)
            
        elif img_type == "noisy":
            text = random.choice(EASY_TEXTS)
            font_size = random.choice([20, 24, 28])
            font_color = get_random_color()
            bg_color = get_random_color(exclude=[font_color])
            noise_level = random.uniform(0.05, 0.1)
            img = create_text_image(text, image_size, image_size//2, font_size, font_color, bg_color, noise_level=noise_level)
        
        filename = f"image_{i}.png"
        img.save(os.path.join(input_dir, filename))
        
        # Store ground truth
        ground_truth[filename] = {
            "text": text,
            "type": img_type,
            "difficulty": "easy" if img_type in ["easy"] else "medium" if img_type in ["medium", "handwritten", "rotated"] else "hard"
        }
    
    # Write ground truth
    gt_path = os.path.join(output_dir, "ground_truth_text.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    # Also create targets.json for compatibility
    targets_path = os.path.join(input_dir, "targets.json")
    with open(targets_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Generated {num_images} OCR test images in {input_dir}")
    print(f"Ground truth saved to {gt_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.dirname(__file__))
    parser.add_argument("--n", type=int, default=15, help="number of images")
    parser.add_argument("--size", type=int, default=512, help="image size")
    args = parser.parse_args()
    generate_dataset(args.out, args.n, args.size)
