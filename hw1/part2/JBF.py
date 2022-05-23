from curses import window
import numpy as np
import cv2
import time

def G_s(p,q,sigma):
    return np.exp(-(((p[0]-q[0])**2+(p[1]-q[1])**2)/(2*sigma^2)))
def G_r(T,p,q,sigma):
    if len(T.shape) == 3:
        return np.exp(-(((T[p[0],p[1],0]-T[q[0],q[1],0])**2+(T[p[0],p[1],1]-T[q[0],q[1],1])**2)+(T[p[0],p[1],2]-T[q[0],q[1],2])**2/(2*sigma**2)))
    else:
        return np.exp(-(((T[p[0],p[1]]-T[q[0],q[1]])**2)/(2*sigma**2)))
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
        T = padded_guidance/255       
        output = np.zeros(img.shape)
        # for k in range(output.shape[2]):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                y, x = np.mgrid[-self.pad_w:self.pad_w+1, -self.pad_w:self.pad_w+1]
                ker_s = np.exp(-(y**2+x**2)/(2*((self.sigma_s)**2)))
                if len(guidance.shape)==2:
                    ker_r = np.exp(-((np.ones((self.wndw_size,self.wndw_size))*T[self.pad_w+i,self.pad_w+j]-T[i:2*self.pad_w+1+i,j:2*self.pad_w+1+j])**2)/(2*self.sigma_r**2))
                else:
                    ker_r = np.exp(-((np.ones((self.wndw_size,self.wndw_size))*T[self.pad_w+i,self.pad_w+j,0]-T[i:2*self.pad_w+1+i,j:2*self.pad_w+1+j,0])**2)/(2*self.sigma_r**2)) * \
                        np.exp(-((np.ones((self.wndw_size,self.wndw_size))*T[self.pad_w+i,self.pad_w+j,1]-T[i:2*self.pad_w+1+i,j:2*self.pad_w+1+j,1])**2)/(2*self.sigma_r**2)) * \
                        np.exp(-((np.ones((self.wndw_size,self.wndw_size))*T[self.pad_w+i,self.pad_w+j,2]-T[i:2*self.pad_w+1+i,j:2*self.pad_w+1+j,2])**2)/(2*self.sigma_r**2))
                ker = np.multiply(ker_s,ker_r)
                ker = ker/ker.sum()
                output[i,j,:] = np.multiply(np.dstack((ker,ker,ker)), padded_img[i:i+2*self.pad_w+1,j:j+2*self.pad_w+1,:]).sum(axis=0).sum(axis=0)
                    
        return np.clip(output, 0, 255).astype(np.uint8)