"""
Uses previously generated CV results from Google Vision API and converts them to text embeddings using a
simple narrative format that pulls in the CV results.

Inspired by https://github.com/daveshap/MultiDocumentAnswering
"""
from typing import Dict, List
import openai
import json
from pathlib import Path
import io

from PIL import Image
from numpy import ndarray
from sentence_transformers import SentenceTransformer


def read_file(fpath:Path) -> str:
    with open(fpath, 'r', encoding='utf-8') as f:
        return f.read()

openai.api_key = read_file('openaiapikey.txt')



def tags_to_embedding(img_meta:Dict) -> (str,List[float]):
    print(f"calculating embeddings")
    prompt = "This picture contains "
    fas = img_meta['face_annotations']
    if len(fas) == 0:
        prompt += "no faces. "
    else:
        prompt += f"{len(fas)} faces. "
        for idx, fa in enumerate(fas):
            face_attributes = []
            for key in fa.keys():
                if key.endswith("_likelihood") and fa[key] >= 4:
                    face_attributes.append(key[0:key.tag_index('_')])
            if len(face_attributes) == 0:
                # prompt += f"Face {idx} is not expressing joy, sorrow, anger or surprise. "
                pass
            else:
                prompt += f"Face {idx} has the following attributes: {','.join(face_attributes)}. "
    labels = img_meta['label_annotations']
    prompt += f"This image contains: {', '.join([i['description'] for i in labels])}. "
    print(f"prompt > {prompt}")
    return prompt, get_embedding(prompt)

def prompt_to_gpt_embedding(prompt:str) -> str:
    print(f"calculating embeddings for GPT description")
    prompt = "describe the following image. " + prompt
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine='text-davinci-002',
                temperature=0.6,
                top_p=1.0,
                max_tokens=2000,
                frequency_penalty=0.25,
                presence_penalty=0.0,
                stop=['<<END>>'],
                prompt=prompt,
            )
            text = response['choices'][0]['text'].strip()
            return text
        except Exception as e:
            print('exception caught:', e)


def get_embedding(prompt, model:str = 'text-similarity-ada-001'):
    content = prompt.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=model)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def generate_tag_index(tag_file:Path, index_file:Path):
    tag_data = {}
    with io.open(tag_file, "r") as tf:
        text = tf.read()
        tag_data = json.loads(text)
    for key in tag_data:
        prompt, embedding = tags_to_embedding(tag_data[key])
        tag_data[key]["embedding"] = embedding
        tag_data[key]["prompt"] = prompt
        tag_data[key]['narrative'] = prompt_to_gpt_embedding(prompt)
        tag_data[key]['narrative_embedding'] = get_embedding(tag_data[key]['narrative'])
    print("writing embeddings to index")
    with io.open(index_file, "w") as outfile:
        outfile.write(json.dumps(tag_data))

def generate_CLIP_index(img_dir:Path) -> Dict:
    print(f"generating CLIP index with data in {img_dir}")
    index = {}
    model = SentenceTransformer('clip-ViT-B-32')
    for file in img_dir.iterdir():
        if file.is_file() and file.suffix in [".jpeg",".jpg"]:
            print(f" calculating embedding for {file}")
            img_emb:ndarray = model.encode(Image.open(file))
            index[file.as_posix()] = {}
            index[file.as_posix()]['embedding'] = img_emb.tolist()
        elif file.is_dir():
            index.update(generate_CLIP_index(file))
    return index

if __name__ == '__main__':
    # generate_tag_index(Path("./data/google_tag_data.txt"), Path("./tag_idx.txt"))

    # index = generate_CLIP_index(Path("/Users/dcripe/dev/ai/gpt/playground/gpt-search/data/"))
    index = generate_CLIP_index(Path("../data/"))
    print("writing embeddings to index")
    with io.open(Path("clip_idx.txt"), "w") as outfile:
        outfile.write(json.dumps(index))

    # index = generate_CLIP_index(Path("/Volumes/photo/image_originals_export/2022/"))
    # print("writing embeddings to index")
    # with io.open(Path("./clip_all_2022_idx.txt"), "w") as outfile:
    #     outfile.write(json.dumps(index))
