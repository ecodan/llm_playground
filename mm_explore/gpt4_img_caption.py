import base64
import json
from pathlib import Path

import requests
from dotenv import load_dotenv, find_dotenv, dotenv_values

load_dotenv(find_dotenv())
config = dotenv_values(find_dotenv())

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your images
ALLOWED_IMAGE_TYPES = ['.jpg','.jpeg','.png']

# single image
# images = ["/Users/dcripe/Pictures/ai/semantic/AmazonPhotos/nik_20120627_174421.jpg"]

# multiple images
images = Path("/Users/dcripe/Pictures/ai/semantic/2022/20220718 - europe:africa/06 Sudtirol").iterdir()

for image_path in images:
    if image_path.suffix not in ALLOWED_IMAGE_TYPES:
        continue

    # check if already captioned
    caption_file = Path(image_path.parent,f'{image_path.name}.caption.json')
    if caption_file.exists():
        with open(caption_file, 'r') as capfile:
            gpt_dump = json.load(capfile)
            # print(gpt_dump)
            if 'choices' in gpt_dump:
                print(f"{image_path.name} already captioned; skipping.")
                continue

    print(f"captioning image {image_path}...")

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['OPENAI_API_KEY']}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image? Answer in 100 words or less."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(f"finished processing {image_path.name}; limit remaining: {response.headers['x-ratelimit-remaining-requests']}")
    if int(response.headers['x-ratelimit-remaining-requests']) <= 0:
        print("exceeded rate limit. stopping.")
        break
    with open(Path(image_path.parent,f'{image_path.name}.caption.json'), 'w') as f:
        json.dump(response.json(), f)

# print(response.json())

'''
High Res 300
{'id': 'chatcmpl-8Jqt6KJYeEbKCs75vaeY5uX0hpesC', 'object': 'chat.completion', 'created': 1699742224, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 1118, 'completion_tokens': 149, 'total_tokens': 1267}, 'choices': [{'message': {'role': 'assistant', 'content': "This image shows two young children enjoying a ride in a gondola, which is a traditional Venetian rowing boat. The children appear to be a young girl on the left, wearing a pink hat and a pink top, and a young boy on the right, holding a water bottle. The gondola has ornate decorations and plush seating, which is typical for these boats that offer romantic and sightseeing experiences in Venice's iconic canals. In the background, we can see the picturesque architecture of Venice, including historic buildings with windows that feature arches and shutters, a small bridge crossing the canal, and the clear blue sky above. The overall ambiance suggests a pleasant and leisurely exploration of the city's waterways."}, 'finish_details': {'type': 'stop', 'stop': '<|fim_suffix|>'}, 'index': 0}]}
'''

'''
Low Res 300
{'id': 'chatcmpl-8Jr2kmVp2J9mCyqBSERovHBBNrc81', 'object': 'chat.completion', 'created': 1699742822, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 98, 'completion_tokens': 132, 'total_tokens': 230}, 'choices': [{'message': {'role': 'assistant', 'content': "This image shows two children sitting in a gondola, a traditional Venetian flat-bottomed boat. They seem to be enjoying a ride through one of the narrow canals of Venice, Italy. The canal is flanked by old buildings with classic European architecture, and there's a bridge crossing the canal in the background. The gondola has ornate metal decorations and plush seating, characteristics of these iconic boats. The children are casually dressed for what appears to be a warm day, and the boy is holding a water bottle. The sun is shining, creating highlights and shadows in the scene, adding to the ambiance of a sunny day in Venice."}, 'finish_details': {'type': 'stop', 'stop': '<|fim_suffix|>'}, 'index': 0}]}
'''

'''
Low Res 100
{'id': 'chatcmpl-8Jr5hUHhOhG6W46wy1fm0ciSD6p8A', 'object': 'chat.completion', 'created': 1699743005, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 106, 'completion_tokens': 97, 'total_tokens': 203}, 'choices': [{'message': {'role': 'assistant', 'content': "The image shows two young children, a girl and a boy, sitting in the ornate interior of a Venetian gondola, a traditional, narrow Italian boat. They are on a canal, with buildings closely lining the sides, and a bridge spans the canal in the background. The children are smiling, and the boy is holding a water bottle. Bright sunlight fills the scene, indicating a clear day, perfect for a gondola ride through Venice's historic waterways."}, 'finish_details': {'type': 'stop', 'stop': '<|fim_suffix|>'}, 'index': 0}]}
'''