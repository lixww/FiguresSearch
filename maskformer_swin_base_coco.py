# Try facebook/maskformer-swin-base-coco on segmenting persons in an image, 
# extract key features from segmentations, compare them across images containing persons.

# MaskFormer, swin backbone, base-sized version, COCO panoptic segmentation
#%%
import torch
import numpy as np
from torch.nn.functional import cosine_similarity

# %%
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("test_images/IMG_4458.png")
image1 = Image.open("test_images/IMG_4339.png")

# %%
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation

# load MaskFormer fine-tuned on COCO panoptic segmentation
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
segmentation = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

#%%
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = segmentation(**inputs)

#%%
# model predicts class_queries_logits of shape `(batch_size, num_queries, num_labels + 1)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

print(class_queries_logits.shape)
print(masks_queries_logits.shape)

#%%
# you can pass them to feature_extractor for postprocessing
result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]], label_ids_to_fuse=set(range(1, 133)))
# we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
print(len(result))
for r in result:
    predicted_panoptic_map = r["segmentation"]
    print(predicted_panoptic_map.shape)
    print(r["segments_info"])
    plt.imshow(predicted_panoptic_map)
    plt.show()

# %%
print(np.array(image).shape)
#%%
person_segments_id = [seg["id"] for seg in result[0]["segments_info"] if seg["label_id"] == 0]
print(person_segments_id)
# %%
def person_overlayed(origin_image, segmentation, person_segments_id):
    origin = np.array(origin_image)
    overlay = origin.copy()

    fig, axes = plt.subplots(1, len(person_segments_id))

    for i, mask_id in enumerate(person_segments_id):
        mask = np.isin(segmentation, mask_id)
        overlay[:,:,0] = np.where(mask == 1, origin[:,:,0], 255)
        overlay[:,:,1] = np.where(mask == 1, origin[:,:,1], 255)
        overlay[:,:,2] = np.where(mask == 1, origin[:,:,2], 255)

        if isinstance(axes, np.ndarray):
            axes[i].imshow(overlay)
            axes[i].set_title(f'Segment id {person_segments_id[i]}')
        else:
            axes.imshow(overlay)
            axes.set_title(f'Segment id {person_segments_id[i]}')
        
    
    # for ax in axes:
    #     ax.axis('off')
    plt.show()

#%%
person_overlayed(image, result[0]["segmentation"], person_segments_id)
#%%
inputs1 = feature_extractor(images=image1, return_tensors="pt")
outputs1 = segmentation(**inputs1)

result1 = feature_extractor.post_process_panoptic_segmentation(outputs1, target_sizes=[image1.size[::-1]], label_ids_to_fuse=set(range(1, 133)))
print(len(result1))

person_segments_id1 = [seg["id"] for seg in result1[0]["segments_info"] if seg["label_id"] == 0]
print(person_segments_id1)
#%%
person_overlayed(image1, result1[0]["segmentation"], person_segments_id1)
# %%
# predict pooling using vit_base_patch_224_in21k
from transformers import ViTImageProcessor, ViTModel
# load model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# %%
def person_overlayed_similarity(backbone_image, other_image, backbone_segmentation, other_segmentation,
                                backbone_person_segments_id, other_person_segments_id):
    backbone_origin = np.array(backbone_image)
    backbone_overlay = backbone_origin.copy()
    other_origin = np.array(other_image)    
    other_overlay = other_origin.copy()

    fig, axes = plt.subplots(len(other_person_segments_id), 1)

    #assuming 1 backbone
    backbone_mask = np.isin(backbone_segmentation, backbone_person_segments_id[0])
    backbone_overlay[:,:,0] = np.where(backbone_mask == 1, backbone_origin[:,:,0], 255)
    backbone_overlay[:,:,1] = np.where(backbone_mask == 1, backbone_origin[:,:,1], 255)
    backbone_overlay[:,:,2] = np.where(backbone_mask == 1, backbone_origin[:,:,2], 255)
    # pooling output
    b_inputs = processor(images=backbone_overlay, return_tensors="pt") # return in pytorch format
    b_outputs = model(**b_inputs)

    for i, mask_id in enumerate(other_person_segments_id):
        mask = np.isin(other_segmentation, mask_id)
        other_overlay[:,:,0] = np.where(mask == 1, other_origin[:,:,0], 255)
        other_overlay[:,:,1] = np.where(mask == 1, other_origin[:,:,1], 255)
        other_overlay[:,:,2] = np.where(mask == 1, other_origin[:,:,2], 255)

        # pooling output
        inputs = processor(images=other_overlay, return_tensors="pt") # return in pytorch format
        outputs = model(**inputs)
        # similarity
        similarity_score = cosine_similarity(b_outputs.pooler_output, outputs.pooler_output)
        print(similarity_score)

        if isinstance(axes, np.ndarray):
            axes[i].imshow(other_overlay)
            axes[i].set_title(f'Segment id {person_segments_id[i]}, similarity_score {similarity_score}')
        else:
            axes.imshow(other_overlay)
            axes.set_title(f'Segment id {person_segments_id[i]}, similarity_score {similarity_score}')
        

# %%
person_overlayed_similarity(image1, image, result1[0]["segmentation"], result[0]["segmentation"],
                            person_segments_id1, person_segments_id)
# %%
