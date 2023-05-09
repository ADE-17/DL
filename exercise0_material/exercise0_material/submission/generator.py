import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import math

class ImageGenerator:
    def _get_corner_points(self, image):
        return image[:, [0, -1], :, :][:, :, [0, -1], :]
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        #Initialising variables 
        self.batch_size = batch_size #int
        self.image_size = image_size #int
        self.rotation = rotation #Bool
        self.mirroring = mirroring #Bool
        self.shuffle = shuffle #Bool
        
        #Load images
        self.list_of_images = [] 
        for image_filename in os.listdir(file_path):
            self.list_of_images.append(os.path.join(file_path, image_filename))
            
        if self.shuffle: #If true, applies a random shuffle within the batch
            np.random.shuffle(self.list_of_images)
            
        #Load labels from JSON   
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        #Computation for epochs
        self.no_of_img_in_batch = len(self.list_of_images)
        self.num_batches_per_epoch = int(np.ceil(self.no_of_img_in_batch / float(self.batch_size)))
        self.index = 0
        self.batches_completed = 0
        
    def next(self):
            
        start_index = self.index
        end_index = min(self.index + self.batch_size, len(self.list_of_images))
        idx = range(start_index, end_index)
            
        images = []
        labels = []
            
        for i in idx:
            image = np.load(self.list_of_images[i])
            image = np.resize(image, self.image_size)
            image = self.augment(image)
            images.append(image)
                
            image_filename = os.path.basename(self.list_of_images[i])
            label_name = self.labels[image_filename.replace('.npy','')]
            # label = self.class_dict[label_name]
            labels.append(label_name)
                
        self.batches_completed += 1
        self.index += self.batch_size
        
        if self.index >= len(self.list_of_images):
            self.index = 0
            self.batches_completed = 0
            if self.shuffle: #If true, Random shuffle within the batch
                np.random.shuffle(self.list_of_images)
                
        if end_index == len(self.list_of_images):             
            images_left = self.batch_size - len(images)
            end_index = images_left
            idx = range(0, end_index)
            for i in idx:
                image = np.load(self.list_of_images[i])
                image = np.resize(image, self.image_size)
                image = self.augment(image)
                images.append(image)
                
                image_filename = os.path.basename(self.list_of_images[i])
                label_name = self.labels[image_filename.replace('.npy','')]
                # label = self.class_dict[label_name]
                labels.append(label_name)

            self.batches_completed += 1
            self.index = images_left
            if self.shuffle: #If true, Random shuffle within the batch
                np.random.shuffle(self.list_of_images)

        images = np.array(images)
        labels = np.array(labels)

        return images, labels
    
    #Function to return current epoch
    def current_epoch(self):
        # number of epochs = batches completed / number of batches in a epoch
        current_epoch = self.batches_completed // self.num_batches_per_epoch
        return current_epoch
    
    #Function to perform augmentation
    def augment(self, image):
        # If true, performs 0, 90, 180, 270 deg rotation at random
        if self.rotation:
            random_angle = np.random.choice([0, 90, 180, 270]) #chose any angle at random to perform a rotation
            image = np.rot90(image, k=random_angle // 90, axes=(0, 1))
        
        #If true, performs mirroring at random
        if self.mirroring:
            if np.random.choice([True, False]): #chose weather to flip a image at random
                image = np.fliplr(image) #using np instead of cv2, simpler.
                
        return image

    def class_name(self, label):
        return self.class_dict[label]
    
    def show(self):
        images, labels = self.next()
        rows = math.ceil(self.batch_size / 5)
        fig, axs = plt.subplots(rows, 5, figsize=(10, 6*rows))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(self.batch_size):
            row = i // 5
            col = i % 5
            axs[row, col].imshow(images[i])
            axs[row, col].set_title(self.class_name(labels[i]))
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
        plt.show()
