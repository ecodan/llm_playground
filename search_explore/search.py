from typing import Dict, List
import io
import openai
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

import make_index


openai.api_key = make_index.read_file('openaiapikey.txt')

model = SentenceTransformer('clip-ViT-B-32')


def calc_similarity(e1, e2):
    return np.dot(e1, e2)


def search_tag_index(text:str, index:Dict, max:int=5, use_narrative:bool=False) -> List:
    print(f"searching index for: '{text}'")
    vector = make_index.get_embedding(text)
    similarities = []
    for key in index.keys():
        img = index[key]
        if use_narrative:
            similarity = calc_similarity(vector, img['narrative_embedding'])
            similarities.append({'content': img['narrative'], 'similarity': similarity, 'img': key})
        else:
            similarity = calc_similarity(vector, img['embedding'])
            similarities.append({'content': img['prompt'], 'similarity': similarity, 'img': key})
    ordered = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    return ordered[0:max]

def search_CLIP_index(text:str, index:Dict, max:int=5) -> List:
    print(f"searching CLIP index for: '{text}'")
    text_emb = model.encode(text)
    similarities = []
    for key in index.keys():
        cos_scores = util.cos_sim(index[key]['embedding'], text_emb)
        similarities.append({
            'content': "",
            'similarity': cos_scores[0],
            'img': key,
            'embedding': index[key]['embedding'],
        })
    ordered = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    return ordered[0:max]

def get_CLIP_pairing(embedding: List[float], index:Dict, max:int=5) -> List:
    print(f"searching CLIP index for pairing")
    similarities = []
    for key in index.keys():
        cos_scores = util.cos_sim(index[key]['embedding'], embedding)
        if cos_scores > 0.999:
            # probably same image, so exclude
            continue
        similarities.append({
            'content': "",
            'similarity': cos_scores[0],
            'img': key,
            'embedding': index[key]['embedding'],
        })
    ordered = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    return ordered[0:2]

def generate_html(query:str, results:List, fname:str ):
    template = None
    with io.open("tplt.html", "r") as f:
        template = f.read()
    title = query[0:len(query) if len(query) < 10 else 10]
    template = template.replace("[[[title]]]", title)
    template = template.replace("[[[prompt]]]", query)
    for idx, r in enumerate(results):
        # template = template.replace(f"[[[img0{idx+1}]]]", f"../data/{r['img']}")
        template = template.replace(f"[[[img0{idx+1}]]]", f"{r['img']}")
        template = template.replace(f"[[[img0{idx+1}score]]]", f"{r['similarity']}")
        template = template.replace(f"[[[img0{idx+1}cont]]]", f"{r['content']}")
    with io.open(f"out/{fname}.html", "w") as outfile:
        outfile.write(template)

if __name__ == '__main__':
    with io.open('../embedding_idx.txt', 'r') as idx:
        tag_index = json.load(idx)
    with io.open('clip_idx.txt', 'r') as idx:
    # with io.open('clip_all_2022_idx.txt', 'r') as idx:
        clip_index = json.load(idx)
    while True:
        query = input(">: ")
        if query == "q":
            print("shutting down...")
            break
        # results = search_tag_index(query, tag_index, use_narrative=False)
        # narrative_results = search_tag_index(query, tag_index, use_narrative=True)
        clip_results = search_CLIP_index(query, clip_index)
        # for r in results:
        #     print(f" img {r['img']}: {r['similarity']} ({r['content']})")
        # for r in narrative_results:
        #     print(f" img {r['img']}: {r['similarity']} ({r['content']})")
        for r in clip_results:
            print(f" img {r['img']}: {r['similarity']} ({r['content']})")
        # generate_html(query, results, fname=f"pr_{query[0:len(query) if len(query) < 10 else 10]}")
        # generate_html(query, narrative_results, fname=f"nr_{query[0:len(query) if len(query) < 10 else 10]}")
        generate_html(query, clip_results, fname=f"cl_{query[0:len(query) if len(query) < 10 else 10]}")