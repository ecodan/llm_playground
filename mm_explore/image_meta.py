import json
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS


IMG_TYPES = ['.jpg','.jpeg','.png']
KEY_FNAME = "filename"
KEY_PARENT = "parent_path"
KEY_DATETIME = "datetime"
KEY_LAT = "lat"
KEY_LON = "lon"
KEY_PEOPLE = "people"
KEY_CAPTION = "caption"
KEY_ASPECT = "aspect"
KEY_HEIGHT = "height"
KEY_WIDTH = "width"

dir = Path('/Users/dcripe/Pictures/ai/semantic/2022/20220718 - europe:africa/06 Sudtirol')
nodes = {}

for f in dir.iterdir():
    if f.suffix in IMG_TYPES:
        print(f"processing {f.name}")
        if f.name not in nodes:
            nodes[f.name] = {}
            nodes[f.name][KEY_FNAME] = f.name
            nodes[f.name][KEY_PARENT] = str(f.parent)
            nodes[f.name][KEY_DATETIME] = ""
            nodes[f.name][KEY_LAT] = ""
            nodes[f.name][KEY_LON] = ""
            nodes[f.name][KEY_PEOPLE] = []
            nodes[f.name][KEY_CAPTION] = ""
            nodes[f.name][KEY_ASPECT] = ""
            nodes[f.name][KEY_HEIGHT] = -1
            nodes[f.name][KEY_WIDTH] = -1

        node = nodes[f.name]

        img = Image.open(f)

        # Get image size
        width, height = img.size
        node[KEY_HEIGHT] = height
        node[KEY_WIDTH] = width
        node[KEY_ASPECT] = "landscape" if width > height else "portrait"

        # Get datetime original taken
        exifdata = img.getexif()
        datetime_original = exifdata.get(36867)
        if datetime_original:
            print(f'Date Taken: {datetime_original}')
            node[KEY_DATETIME] = datetime_original
        else:
            bt = os.stat(str(f)).st_birthtime
            node[KEY_DATETIME] = str(datetime.fromtimestamp(bt))

        # Get geolocation if available
        if 34853 in exifdata:
            lat, lon = exifdata[34853]
            print(f'Latitude: {lat}')
            print(f'Longitude: {lon}')
            node[KEY_LAT] = lat
            node[KEY_LON] = lon


        # add caption if present
        caption_file = Path(f.parent,f'{f.name}.caption.json')
        if caption_file.exists():
            with open(caption_file, 'r') as capfile:
                gpt_dump = json.load(capfile)
                print(gpt_dump)
                if 'choices' in gpt_dump:
                    node[KEY_CAPTION] = gpt_dump['choices'][0]['message']['content']

        with open(Path(f.parent,f'{f.name}.meta.json'), 'w') as outfile:
            json.dump(node, outfile)

