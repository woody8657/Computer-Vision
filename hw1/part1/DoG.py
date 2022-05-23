import numpy as np
import cv2

def convolution(img, kernel):
    filter_size = kernel.shape[0]
    img_padded = np.pad(img,  [(int((filter_size-1)/2),int((filter_size-1)/2)),(int((filter_size-1)/2),int((filter_size-1)/2))], 'symmetric')
    result = np.zeros(img.shape)
    y_start, y_end = int((filter_size-1)/2), int(img.shape[0]+(filter_size-1)/2)
    x_start, x_end = int((filter_size-1)/2), int(img.shape[1]+(filter_size-1)/2)
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            result[i-y_start, j-x_start] = np.sum(np.multiply(img_padded[i-y_start:i+y_start+1, j-y_start:j+y_start+1],kernel))

    return result

def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 255
    img.astype(np.uint8)
    
    return img

def get_filter(sigma, img_num):
    k = 2 * int(4*sigma**img_num+0.5)+1
    i, j = np.mgrid[-(k-1)/2:(k-1)/2+1, -(k-1)/2:(k-1)/2+1]
    ker = np.exp(-(i**2+j**2)/(2*((sigma**img_num)**2)))
    ker = ker / ker.sum()
    
    return ker



class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        
        # count = 0
        for i in range(self.num_octaves):
            gaussian_images.append(image)
            for j in range(self.num_guassian_images_per_octave-1):
                tmp = convolution(image, get_filter(self.sigma,j+1))
                gaussian_images.append(tmp)
            image = cv2.resize(tmp, (int(image.shape[1]/2),int(image.shape[0]/2)), interpolation=cv2.INTER_NEAREST)
      
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        # count = 0
        for i in range(len(gaussian_images)):
            if (i % self.num_guassian_images_per_octave) != self.num_DoG_images_per_octave:
                dog_images.append(gaussian_images[i+1]-gaussian_images[i])
                # tmp = normalize(gaussian_images[i+1]-gaussian_images[i])
                # cv2.imwrite(str(count)+'.png', tmp)
                # count = count+1

        

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints_list = [] 
        for n in range(1,len(dog_images)-1):
            if (dog_images[n].shape==dog_images[n-1].shape) and (dog_images[n].shape==dog_images[n+1].shape):
                scale = dog_images[0].shape[0] / dog_images[n].shape[0]
                for i in range(1,dog_images[n].shape[0]-1):
                    for j in range(1,dog_images[n].shape[1]-1):
                        cube = np.stack((dog_images[n-1][i-1:i+2,j-1:j+2],dog_images[n][i-1:i+2,j-1:j+2],dog_images[n+1][i-1:i+2,j-1:j+2]), axis=2)
                        if ((dog_images[n][i,j] == cube.max()) or (dog_images[n][i,j] == cube.min())) and (abs(cube[1,1,1])>self.threshold):
                            keypoints_list.append([scale*i,scale*j])
            
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.array(keypoints_list)
        keypoints = np.unique(keypoints, axis=0)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints

