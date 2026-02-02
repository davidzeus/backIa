import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from dotenv import load_dotenv
load_dotenv(override=True)

from app.services.image_analysis_service import analyze_medical_image

async def main():
    # Use the existing image found in project root
    image_path = "a7f8a5e4-582e-4faf-bb97-50b9e7181ebb.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}. Please place a test image there.")
        return

    print(f"Reading image: {image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print("Sending to Ollama... please wait.")
    try:
        description = await analyze_medical_image(image_bytes, prompt="Describe detalladamente qué se observa en esta imagen médica.")
        print("\nAnalysis Result:\n")
        print(description)
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())
