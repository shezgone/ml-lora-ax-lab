import argparse
import sys
from mlx_vlm import load, generate
from mlx_vlm.utils import load_image

def analyze_blueprint(image_path, prompt="이 도면을 자세히 설명해줘.", model_path="mlx-community/paligemma-3b-mix-448-8bit"):
    """
    PaliGemma를 사용하여 도면 이미지를 분석합니다.
    """
    print(f"Loading model: {model_path}...")
    model, processor = load(model_path)
    
    try:
        image = load_image(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    formatted_prompt = f"<image>{prompt}"
    
    print(f"Analyzing image: {image_path}...")
    print(f"Prompt: {prompt}")
    print("-" * 30)
    
    response = generate(
        model, 
        processor, 
        image, 
        formatted_prompt, 
        max_tokens=500, 
        verbose=True
    )
    
    print("-" * 30)
    print("Analysis Complete.")

if __name__ == "__main__":
    # 사용법: python infer_blueprint.py --image ./my_drawing.png --prompt "이 도면의 치수를 알려줘"
    parser = argparse.ArgumentParser(description="Analyze blueprints using PaliGemma on MLX")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Prompt for the model")
    parser.add_argument("--model", type=str, default="mlx-community/paligemma-3b-mix-448-8bit", help="Model path (HuggingFace)")
    
    args = parser.parse_args()
    
    analyze_blueprint(args.image, args.prompt, args.model)
