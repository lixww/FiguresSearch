# ViT model, pre-trained, 224x224 patch size, ImageNet-21k.
# %%
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("test_images/IMG_4458.png")

# %%
# preview the image 
plt.imshow(image)
plt.show()

# %%
# load model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
# %%
inputs = processor(images=image, return_tensors="pt") # return in pytorch format
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
# %%
print(last_hidden_states)
# %%
image1 = Image.open("test_images/IMG_4339.png")
inputs1 = processor(images=image1, return_tensors="pt")
outputs1 = model(**inputs1)
last_hidden_states1 = outputs1.last_hidden_state
# %% 
print(last_hidden_states1.shape)

# %%
from torch.nn.functional import cosine_similarity

pooler_outputs = outputs.pooler_output
pooler_outputs1 = outputs1.pooler_output
similarity_score = cosine_similarity(pooler_outputs, pooler_outputs1)
print(similarity_score)
# %%
print(outputs.pooler_output.shape)
# %%
