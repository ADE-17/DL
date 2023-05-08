import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def _get_corner_points(self, image):
        # Utility function to check whether the augmentations where performed
        # expects batch of image - expected shape is [s,x,y,c]
        return image[:, [0, -1], :, :][:, :, [0, -1], :]
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        
        self.image_filenames = []
        for filename in os.listdir(file_path):
            self.image_filenames.append(os.path.join(file_path, filename))
            
        if self.shuffle:
            np.random.shuffle(self.image_filenames)
            
        #Labels 
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        # self.class_indices = {}
        # for i, class_name in self.class_dict.items():
        #     self.class_indices[class_name] = i
        self.num_samples = len(self.image_filenames)
        self.num_batches_per_epoch = int(np.ceil(self.num_samples / float(self.batch_size)))
        self.index = 0
        self.num_batches_completed = 0
        
    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        
        start_index = self.index
        end_index = min(self.index + self.batch_size, len(self.image_filenames))
        indices = range(start_index, end_index)
        
        images = []
        labels = []
        
        for i in indices:
            image = np.load(self.image_filenames[i])
            image = np.resize(image, self.image_size)
            images.append(image)
            
            filename = os.path.basename(self.image_filenames[i])
            label_name = self.labels[filename.replace('.npy','')]
            label = self.class_dict[label_name]
            labels.append(label)
            
        self.num_batches_completed += 1
        self.index += self.batch_size
        if self.index >= len(self.image_filenames):
            self.index = 0
            if self.shuffle:
                np.random.shuffle(self.image_filenames)
                
        if end_index == len(self.image_filenames):             
            images_remaining = self.batch_size - len(images)
            end_index = images_remaining
            indices = range(0, end_index)
            for i in indices:
                image = np.load(self.image_filenames[i])
                image = np.resize(image, self.image_size)
                images.append(image)
                
                filename = os.path.basename(self.image_filenames[i])
                label_name = self.labels[filename.replace('.npy','')]
                label = self.class_dict[label_name]
                labels.append(label)
            
            # Update the index for the next batch
            self.num_batches_completed += 1
            self.index = images_remaining
            if self.shuffle:
                np.random.shuffle(self.image_filenames)
        # pass
        images = np.array(images)
        labels = np.array(labels)
    
        return images, labels
    
    def current_epoch(self):
        # return the current epoch number
        return self.num_batches_completed // self.num_batches_per_epoch
    
    def augment(self, img):
        # Perform a random rotation of 0, 90, 180 or 270 degrees
        if self.rotation:
            rotation_angle = np.random.choice([0, 90, 180, 270])
            img = np.rot90(img, k=rotation_angle // 90, axes=(0, 1))
            
        if self.mirroring:
        # randomly mirror image horizontally
            if np.random.choice([True, False]):
                img = np.fliplr(img)
                
        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        pass

