"""
Tags a set of images with Google Vision API and writes output to JSON.

Inspired by https://github.com/daveshap/MultiDocumentAnswering
"""
from typing import Dict
import json
from pathlib import Path
from google.cloud import vision
import io


def tag_photos(datadir: Path, outfile_name: str = "google_tag_data.txt") -> Dict:
    print(f"tagging images in {datadir}")
    client = vision.ImageAnnotatorClient()
    ftags = {}
    flist = [
        vision.Feature.Type.LABEL_DETECTION,
        vision.Feature.Type.LANDMARK_DETECTION,
        vision.Feature.Type.FACE_DETECTION,
        vision.Feature.Type.SAFE_SEARCH_DETECTION,
        vision.Feature.Type.IMAGE_PROPERTIES,
    ]
    features = [vision.Feature(type_=f, max_results=100) for f in flist]

    for file in datadir.iterdir():
        if file.is_file() and file.suffix in [".jpeg", ".jpg"]:
            print(f"  tagging image {file.name}")
            with io.open(file, 'rb') as image_file:
                content = image_file.read()
                image = vision.Image(content=content)
                request = vision.AnnotateImageRequest(
                    image=image,
                    features=features
                )
                response: vision.AnnotateImageResponse = client.annotate_image(request=request)
                ftags[file.name] = vision.AnnotateImageResponse.to_dict(response)
    outfile_path = Path(datadir, outfile_name)
    print(f"done tagging images; writing data to {outfile_path}")
    with io.open(outfile_path, "w") as out:
        out.write(json.dumps(ftags))


if __name__ == '__main__':
    tag_photos(Path("../data"))
