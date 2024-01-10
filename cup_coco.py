# %%
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.visualize import display_instances, display_top_masks
from mrcnn.utils import extract_bboxes

from mrcnn.utils import Dataset
from matplotlib import pyplot as plt

from mrcnn.config import Config
from mrcnn.model import MaskRCNN


from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw


# %%
class CocoLikeDataset(utils.Dataset):
    
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids
    

    def merge_dataset(self, other_dataset):
        """ Merge another dataset into the current one. """
        for image_info in other_dataset.image_info:
            self.add_image(
                source=image_info['source'],
                image_id=image_info['id'],
                path=image_info['path'],
                width=image_info['width'],
                height=image_info['height'],
                annotations=image_info['annotations']
            )
        for class_info in other_dataset.class_info:
            if class_info['source'] == 'coco_like' and class_info['id'] != 0:
                self.add_class(
                    class_info['source'],
                    class_info['id'],
                    class_info['name']
                )



# %% [markdown]
# Load Datasets

# %%

# Load and prepare datasets
dataset_train1 = CocoLikeDataset()
dataset_train1.load_data('train\labels\labels_my-project-name_2023-12-26-01-26-46_coco - Copy.json', 'train/100_more')
dataset_train1.prepare()

dataset_train2 = CocoLikeDataset()
dataset_train2.load_data(r'train\label_val\val_test_coco.json', r'train\100_more\100')
dataset_train2.prepare()

# Merge datasets
dataset_train1.merge_dataset(dataset_train2)

#In this example, I do not have annotations for my validation data, so I am loading train data
dataset_val = CocoLikeDataset()
dataset_val.load_data(r'val\labels\val_coco.json', r'val\images')
dataset_val.prepare()


# %%
dataset = dataset_train1
image_ids = dataset.image_ids
#image_ids = np.random.choice(dataset.image_ids, 3)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    #display_top_masks(image, mask, class_ids, dataset.class_names, limit=2)  #limit to total number of classes


# %%
# define image id
image_id = 4
# load the image
image = dataset_train1.load_image(image_id)
# load the masks and the class ids
mask, class_ids = dataset_train1.load_mask(image_id)

# display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
# dataset.class_names, r1['scores'], ax=ax, title="Predictions1")

# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, dataset_train1.class_names)

# %% [markdown]
# Define a configuration for the model

# %%
class CupConfig(Config):
    # define the name of the configuration
    NAME = "cup_cfg_coco"
    # number of classes 
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 20
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    # Adjust this value as needed
    LEARNING_RATE = 0.001

# prepare config
config = CupConfig()
config.display()


# %%
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")


# %%

from tensorflow.keras.callbacks import Callback


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

history = LossHistory()


# %% [markdown]
# Define the model

# %%
# define the model
model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
#model.train(dataset_train, dataset_train, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')

# %% [markdown]
# Initiating the training

# %%
# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE, 
#             epochs=25, 
#             layers='heads', 
#             custom_callbacks=[history])

# Training the model on the combined dataset
model.train(dataset_train1, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=60, 
            layers='heads', 
            custom_callbacks=[history])


# %% [markdown]
# Plots

# %%
plt.figure(figsize=(8, 4))
plt.plot(history.train_losses, label='Train Loss',color='black')
if history.val_losses:
    plt.plot(history.val_losses, label='Validation Loss',color='red')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# Model  Inference

# %% [markdown]
# Importing libraries for Inference

# %%
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib.patches import Rectangle

# %% [markdown]
# Define the prediction configuration and Calculate mAP

# %%
# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "cup_cfg_coco"
	# number of classes (background + Blue Marbles + Non Blue marbles)
	NUM_CLASSES = 1 + 1
	# Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
 
# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id , use_mini_mask=False
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

# %% [markdown]
# Create config and evaluate the dataset

# %%
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
# load model weights
model.load_weights('logs\cup_cfg_coco20240106T1216\mask_rcnn_cup_cfg_coco_0060.h5', by_name=True)

# evaluate model on training dataset
train_mAP = evaluate_model(dataset_train1, model, cfg)
print("Train mAP(Mean Average Precision): %.3f" % train_mAP)



# %%
# evaluate model on test dataset
test_mAP = evaluate_model(dataset_val, model, cfg)
print("Test mAP: %.3f" % test_mAP)

# %% [markdown]
# Test on a single image
# 

# %%
cup_img = skimage.io.imread(r"test\6\71shzwk4sDL.jpg")
plt.imshow(cup_img)

detected = model.detect([cup_img])
# Perform detection
results = model.detect([cup_img], verbose=0)[0]

# Filter detections with a score above 0.99
indices = np.where(results['scores'] > 0.99)[0]
filtered_rois = results['rois'][indices]
filtered_class_ids = results['class_ids'][indices]
filtered_scores = results['scores'][indices]
filtered_masks = results['masks'][:, :, indices]

# Display the filtered results
display_instances(cup_img, filtered_rois, filtered_masks, filtered_class_ids, 
                  class_names, filtered_scores)

# %% [markdown]
# Test on a set of Images

# %%
# Directory containing the images
image_folder = r"test\test_images"  # Replace with the path to your folder

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.webp'))]

# Process each image in the folder
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = skimage.io.imread(image_path)

    # Perform detection using the model
    detected = model.detect([image])
    results = detected[0]
    class_names = ['BG', 'Cylindrical Cup', 'Non_Cylindrical Cup']

    # Filter detections by score threshold (0.99)
    high_score_indices = np.where(results['scores'] > 0.994)[0]
    high_score_rois = results['rois'][high_score_indices]
    high_score_class_ids = results['class_ids'][high_score_indices]
    high_score_scores = results['scores'][high_score_indices]
    high_score_masks = results['masks'][:, :, high_score_indices]

    # Display instances with high scores
    display_instances(image, high_score_rois, high_score_masks, 
                      high_score_class_ids, class_names, high_score_scores)



