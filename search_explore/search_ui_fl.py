import io
import json
from typing import Dict, List
from PIL import Image


from flask import Flask, render_template, request, send_from_directory, send_file

from search import search_CLIP_index, get_CLIP_pairing

app = Flask(__name__)

clip_index = {}
with io.open('clip_idx.txt', 'r') as idx:
# with io.open('clip_all_2022_idx.txt', 'r') as idx:
    clip_index = json.load(idx)

@app.route('/')
def index():
    return render_template('search.html', pictures={})

@app.route('/search/', methods = ['POST'])
def data():
    if request.method == 'POST':
        pictures:List[Dict] = search_CLIP_index(request.form.get("prompt"), index=clip_index)
        for p in pictures:
            image = Image.open(p['img'])
            if image.height > image.width:
                p['pairings'] = get_CLIP_pairing(p['embedding'], index=clip_index)
        return render_template('search.html',pictures = pictures, prompt=request.form.get("prompt"))


@app.route('/img/<path:filename>')
def serve_image(filename):
    print(f"serve_image: {filename}")
    # if not filename.startswith("/"):
    #     filename = "/" + filename
        # print(filename)
    return send_file(filename, mimetype='image/jpeg')

app.run(host='localhost', port=5555)