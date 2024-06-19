#%%
# Prerequists
from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

import shutil

#%%
# reference figure
ref_image = "test_images/IMG_4339.png"

# gallery
gallery = "test_images/"


#%%
def checkTruncatedFiles(path):
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            try:
                img = Image.open(os.path.join(path, filename))
                img.verify()  # verify that it is, in fact, an image
            except (IOError, SyntaxError) as e:
                print(f'The file {filename} is truncated or not an image')
                continue

checkTruncatedFiles(gallery)
#%%
def writeInTxt(list:list, filename:str="identities"):
    
    with open(filename+'.txt', 'w') as f:
         # Iterate over the list
        for item in list:
            # Write each item to the file
            f.write(f"{item}\n")


#%%
# use Facenet 512

result = DeepFace.find(ref_image, gallery, "Facenet512")


#%%
print(f"{len(result)} figures are detected in gallery.")
print("---- ----")

for idx, i in enumerate(result):
    print(f"Figure No.{idx} ->")
    num_entities, num_features = i.shape
    print(f"Number of entities: {num_entities}")
    print(f"Number of features: {num_features}")
    print("----")
    print(i['identity'])
    writeInTxt(i['identity'], f"identities{idx}")
    print("---- ----")
# %%
def copyToDir(list, path):
    for i in list:
        shutil.copy(i, path)
    print("Finished copy-paste, check {path}." )

for idx, i in enumerate(result):
    print(f"Figure No.{idx} ->")
    num_entities, num_features = i.shape
    print(f"Number of entities: {num_entities}")
    print(f"Number of features: {num_features}")
    print("----")
    copyToDir(i["identity"], "test_path")
    print("---- ----")
# %%
