#!/usr/bin/env python3
"""
Demonstration of CNN failure when applied to microscopy images.

This script shows how a ResNet model trained on ImageNet works correctly on natural images
(like cats) but produces confident but completely irrelevant predictions when applied 
to microscopy data.

The microscopy image is from the Human Protein Atlas showing TRAP1 protein localization.
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

def load_imagenet_classes():
    """Load ImageNet class labels."""
    # Download ImageNet class index
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        with urllib.request.urlopen(url) as response:
            classes = [line.decode('utf-8').strip() for line in response.readlines()]
    except:
        # Fallback to a subset of common ImageNet classes
        classes = [
            'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark',
            'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch',
            'house finch', 'junco', 'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie',
            'chickadee', 'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl'
        ] + ['unknown'] * 975  # Pad to 1000 classes
    
    return classes

def preprocess_image(image_path):
    """Preprocess the image for ResNet inference."""
    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    return input_batch, image

def get_predictions(model, input_batch, classes, top_k=3):
    """Get top-k predictions from the model."""
    model.eval()
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = F.softmax(outputs[0], dim=0)
        
        # Get top-k predictions
        top_prob, top_catid = torch.topk(probabilities, top_k)
        
        predictions = []
        for i in range(top_k):
            prob = top_prob[i].item()
            catid = top_catid[i].item()
            class_name = classes[catid] if catid < len(classes) else f"Class_{catid}"
            predictions.append((class_name, prob * 100))
    
    return predictions

def add_predictions_to_image(image, predictions, is_correct=True):
    """Add prediction text overlay to image."""
    # Create a copy to draw on
    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    # Background color - use darker gray for better contrast
    bg_color = (64, 64, 64, 200)  # Darker gray with more opacity
    text_color = (255, 255, 255)  # White text for better contrast on gray
    
    # Create overlay
    overlay = Image.new('RGBA', img_with_text.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Calculate text area
    y_offset = 10
    max_width = 0
    text_lines = []
    
    for i, (class_name, confidence) in enumerate(predictions):
        clean_name = class_name.replace('_', ' ').title()
        if len(clean_name) > 20:
            clean_name = clean_name[:17] + "..."
        text_line = f"{i+1}. {clean_name} ({confidence:.1f}%)"
        text_lines.append(text_line)
        bbox = draw.textbbox((0, 0), text_line, font=font_small)
        max_width = max(max_width, bbox[2] - bbox[0])
    
    # Draw background rectangle
    padding = 8
    rect_height = len(text_lines) * 20 + padding * 2
    overlay_draw.rectangle([(10, y_offset), (10 + max_width + padding * 2, y_offset + rect_height)], 
                          fill=bg_color)
    
    # Draw text
    for i, text_line in enumerate(text_lines):
        overlay_draw.text((10 + padding, y_offset + padding + i * 20), text_line, 
                         fill=text_color, font=font_small)
    
    # Composite overlay onto image
    img_with_text = Image.alpha_composite(img_with_text.convert('RGBA'), overlay).convert('RGB')
    
    return img_with_text

def create_comparison_visualization(cat_image_path, micro_image_path, cat_predictions, micro_predictions, output_path):
    """Create a side-by-side comparison visualization."""
    # Load images
    cat_image = Image.open(cat_image_path).convert('RGB')
    micro_image = Image.open(micro_image_path).convert('RGB')
    
    # Crop cat image to square
    cat_width, cat_height = cat_image.size
    cat_size = min(cat_width, cat_height)
    cat_left = (cat_width - cat_size) // 2
    cat_top = (cat_height - cat_size) // 2
    cat_image = cat_image.crop((cat_left, cat_top, cat_left + cat_size, cat_top + cat_size))
    
    # Resize images to same height for comparison
    target_height = 400
    cat_width = target_height  # Square now
    micro_aspect = micro_image.width / micro_image.height
    micro_width = int(target_height * micro_aspect)
    
    cat_image = cat_image.resize((cat_width, target_height), Image.Resampling.LANCZOS)
    micro_image = micro_image.resize((micro_width, target_height), Image.Resampling.LANCZOS)
    
    # Add predictions to images
    cat_image_with_pred = add_predictions_to_image(cat_image, cat_predictions, is_correct=True)
    micro_image_with_pred = add_predictions_to_image(micro_image, micro_predictions, is_correct=False)
    
    # Create matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display cat image
    ax1.imshow(cat_image_with_pred)
    ax1.set_title('Control: Natural Image\n(ResNet works correctly)', fontsize=14, fontweight='bold', color='black')
    ax1.axis('off')
    
    # Display microscopy image
    ax2.imshow(micro_image_with_pred)
    ax2.set_title('Microscopy Image: TRAP1 Protein\n(ResNet fails dramatically)', fontsize=14, fontweight='bold', color='black')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Add title at the bottom
    fig.text(0.5, 0.02, 'CNN Domain Mismatch: ImageNet vs. Microscopy', 
             ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    return fig

def main():
    print("Loading pretrained ResNet50 model...")
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    print("Loading ImageNet class labels...")
    classes = load_imagenet_classes()
    
    # Paths to images
    cat_image_path = "3-llms-assets/IMG_5593 Medium.jpeg"
    micro_image_path = "3-llms-assets/860_D8_1_blue_red_green.jpg"
    
    print(f"Processing cat image: {cat_image_path}")
    cat_input, _ = preprocess_image(cat_image_path)
    cat_predictions = get_predictions(model, cat_input, classes, top_k=3)
    
    print(f"Processing microscopy image: {micro_image_path}")
    micro_input, _ = preprocess_image(micro_image_path)
    micro_predictions = get_predictions(model, micro_input, classes, top_k=3)
    
    print("\nResults:")
    print("="*60)
    print("Cat image predictions (CORRECT):")
    for i, (class_name, confidence) in enumerate(cat_predictions, 1):
        print(f"  {i}. {class_name.replace('_', ' ').title()}: {confidence:.1f}%")
    
    print("\nMicroscopy image predictions (WRONG):")
    for i, (class_name, confidence) in enumerate(micro_predictions, 1):
        print(f"  {i}. {class_name.replace('_', ' ').title()}: {confidence:.1f}%")
    
    print("\n" + "="*60)
    print("⚠️  The microscopy predictions are completely irrelevant!")
    print("The image shows TRAP1 protein in human cells from the Human Protein Atlas.")
    print("This demonstrates why domain-specific training is essential.")
    
    # Create visualization
    output_path = "3-llms-assets/cnn_microscopy_failure.png"
    print(f"\nCreating comparison visualization...")
    create_comparison_visualization(cat_image_path, micro_image_path, 
                                  cat_predictions, micro_predictions, output_path)
    
    print("Done!")

if __name__ == "__main__":
    main() 