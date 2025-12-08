"""
Predicts on a target image using a trained PyTorch model
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from typing import List
import model_builder  # Assumes you have model_builder.py with TinyVGG class

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = None):
    """Makes a prediction on a target image and plots the result"""
    
    # Load image
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255
    
    # Apply transform if provided
    if transform:
        target_image = transform(target_image)
    
    # Ensure model is on correct device
    model.to(device)
    
    # Make prediction
    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(0)  # Add batch dimension
        target_image_pred = model(target_image.to(device))
    
    # Convert logits to prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    
    # Convert prediction probabilities to prediction label
    target_image_pred_labels = torch.argmax(target_image_pred_probs, dim=1)
    
    # Get max probability as percentage
    max_prob = target_image_pred_probs.max().cpu().item()
    
    return target_image_pred_labels.cpu().item(), max_prob


if __name__ == "__main__": # This line makes it possible to use it as standalone or modular
    # Create argument parser
    parser = argparse.ArgumentParser(description="Predict on a target image using a trained PyTorch model")
    
    # Add arguments
    parser.add_argument("image_path", 
                       type=str, 
                       help="Path to target image file (e.g., pizza.jpg)")
    parser.add_argument("--model_path", 
                       type=str, 
                       default="models/pizza_steak_sushi_model_1.pth",
                       help="Path to trained model")
    parser.add_argument("--image_size", 
                       type=int, 
                       default=64,
                       help="Size to resize image to")
    parser.add_argument("--hidden_units", 
                       type=int, 
                       default=16,
                       help="Number of hidden units in model (must match trained model)")
    parser.add_argument("--class_names", 
                       nargs="+",
                       default=["pizza", "steak", "sushi"],
                       help="List of class names")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define image transform
    image_transform = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size))
    ])
    
    # Load model
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(args.class_names),
        input_size=args.image_size
    ).to(device)
    
    # Load trained weights
    print(f"[INFO] Loading model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Make prediction
    print(f"[INFO] Making prediction on: {args.image_path}")
    pred_class, pred_prob = pred_and_plot_image(
        model=model,
        image_path=args.image_path,
        class_names=args.class_names,
        transform=image_transform,
        device=device
    )
    
    print(f"[INFO] Predicted class: {args.class_names[pred_class]} with a {pred_prob*100:.2f}% probability")