#%%
# Prerequists
from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

#%%
image_path1 = "test_images/IMG_4339.png"
image_path2 = "test_images/IMG_4458.png"

#%%

# result = DeepFace.verify(img1_path=image_path1, img2_path=image_path2) #default VGG-Face
result = DeepFace.verify(img1_path=image_path1, img2_path=image_path2, model_name='Facenet512') #Facenet 512
# %%
print(result)
# %%
# image1 = plt.imread(image_path1) 
# image2 = plt.imread(image_path2)
image1 = Image.open(image_path1)
image2 = Image.open(image_path2)

fig, ax = plt.subplots(1, 2)

ax[0].imshow(image1)
x1 = result['facial_areas']['img1']['x']
y1 = result['facial_areas']['img1']['y']
w1 = result['facial_areas']['img1']['w']
h1 = result['facial_areas']['img1']['h']
rect1 = patches.Rectangle((x1, y1), w1, h1, linewidth=1, edgecolor='g', facecolor='none')
ax[0].add_patch(rect1)

ax[1].imshow(image2)
x2 = result['facial_areas']['img2']['x']
y2 = result['facial_areas']['img2']['y']
w2 = result['facial_areas']['img2']['w']
h2 = result['facial_areas']['img2']['h']
rect2 = patches.Rectangle((x2, y2), w2, h2, linewidth=1, edgecolor='g', facecolor='none')
ax[1].add_patch(rect2)

plt.show()

# %%
