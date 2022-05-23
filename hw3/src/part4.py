from email import iterators
import numpy as np
import cv2
import random
from sklearn.neighbors import RadiusNeighborsRegressor
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None
    H_list = []
    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        
        # TODO: 1.feature detection & matching
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
        
        # create BFMatcher objects
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(im1,kp1,im2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # matches = matches[:10]
        # cv2.imwrite('tmp2.png',img3)
        points1 = np.array([kp1[matches[i].queryIdx].pt for i in range(len(matches))])
        # for x,y in points1:
        #     output1 = cv2.circle(im1, (x,y), 5, (0,0,255), 5) 
        points2 = np.array([kp2[matches[i].trainIdx].pt for i in range(len(matches))])
        # for x,y in points2:
        #     output2 = cv2.circle(im2, (x,y), 5, (0,0,255), 5) 
        # cv2.imwrite('tmp1.png', np.concatenate((output1,output2),axis=1))
        
        # raise
        # TODO: 2. apply RANSAC to choose best H
        threshold = 1
        iteration = 100000
        np.random.seed(42)
        num_inliers = 0
        for i in range(iteration):
            index = np.random.choice(points1.shape[0], 4)
            try:
                tmp_H = solve_homography(points1[index,:], points2[index,:])
            except:
                continue
            tmp_points1 = np.concatenate((points1, np.ones((points1.shape[0],1))),axis=1)
            tmp_points2 = np.concatenate((points2, np.ones((points2.shape[0],1))),axis=1)
            tmp_des = np.dot(tmp_H, tmp_points1.T)
            tmp_des = tmp_des / tmp_des[2,:]
            error = tmp_des[:2,:] - tmp_points2.T[:2,:]
            
            norm = np.linalg.norm(error,axis=0)
        
            tmp_inliers = len(np.where(norm<=threshold)[0])
            if tmp_inliers > num_inliers:
                H = tmp_H
                num_inliers = tmp_inliers
                print(f'{num_inliers} inliers, H updates!!')
        H_list.append(H)        
        # TODO: 3. chain the homographies
        
        if idx>0:
            H = H_list[0]
            for j in range(idx):
                H = np.dot(H_list[j+1],H)
        # TODO: 4. apply warping
        h, w, c = im2.shape
  
        dst = warping(im2, dst, np.linalg.inv(H), 0, h, (idx+1)*w, (idx+2)*w, direction='b')
        out = dst
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)