import numpy as np
import cv2

def conv(img, guide, wndw_size, sigma_s, sigma_r):

    # gaussian_kernel
    d = int((wndw_size-1)/2)
    x, y = np.mgrid[-d:d+1, -d:d+1]
    g_kernel = np.exp(-(x**2+y**2)/(2*(sigma_s**2)))
    
    # null image
    conv_img = np.zeros(img.shape)
    # range_kernel
    if len(guide.shape)<3: #( use guide, 1 channel: gray scale )
        guide = np.dstack([guide]*3)

        for i in range(d, guide.shape[0]-d):
            for j in range(d, guide.shape[1]-d):

                block = guide[i-d:i-d+wndw_size , j-d:j-d+wndw_size , : ] # (wndw_size) x (wndw_size) x 3
                temp = []
                for k in range(3):
                    r_kernel = np.exp(  -1*(np.square(block[:,:,k]/255 - block[d,d,k]/255))/(2*(sigma_r**2))  ) # (wndw_size) x (wndw_size) x 1
                    kernel = g_kernel*r_kernel
                    kernel = kernel/kernel.sum()
                    temp.append(kernel)
                kernel = np.dstack(temp)
                block = img[i-d:i-d+wndw_size , j-d:j-d+wndw_size , : ]

                conv_img[i,j,:] = (kernel*block).sum(axis=(0,1)) # filtering

    

    else:
        for i in range(d, guide.shape[0]-d):
            for j in range(d, guide.shape[1]-d):

                block = guide[i-d:i-d+wndw_size , j-d:j-d+wndw_size , : ] # (wndw_size) x (wndw_size) x 3
                r_kernel = np.exp(  -1*(np.square(block/255 - block[d,d,:]/255).sum(axis=2))/(2*(sigma_r**2))  ) # (wndw_size) x (wndw_size) x 1
                kernel = g_kernel*r_kernel
                kernel = kernel/kernel.sum()
                kernel = np.dstack([kernel]*3)

                conv_img[i,j,:] = (kernel*block).sum(axis=(0,1)) # filtering

    return conv_img[d:-d,d:-d,:]



class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        output = conv(padded_img, padded_guidance, self.wndw_size, self.sigma_s, self.sigma_r)
        
        return np.clip(output, 0, 255).astype(np.uint8)