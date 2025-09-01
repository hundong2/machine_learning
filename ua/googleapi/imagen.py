from basic import get_gemini_client
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

def make_image():
    client = get_gemini_client()
    response = client.models.generate_images(
    model='imagen-4.0-generate-001',
    prompt='Robot holding a red skateboard',
    config=types.GenerateImagesConfig(
        number_of_images= 4,
        )   
    )
    for generated_image in response.generated_images:
        generated_image.image.show()

def main():
    make_image()

if __name__=="__main__": 
    main()