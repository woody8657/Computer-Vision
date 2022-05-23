import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def get_txt(path):
    txt_info = []
    with open(path) as f:
        for count, line in enumerate(f.readlines()):
            s = line.split(',')
            txt_info.append([i.replace('\n','') for i in s])
    RGB = txt_info[1:-1]
    sigma = txt_info[-1]
    return RGB, sigma

def compute_cost(img, RGB, sigma):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for r, g, b in RGB:
        if (float(r),float(g),float(b)) == (0.1, 0, 0.9):
            # guidance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            guidance = img_rgb[:,:,0] * float(r) + img_rgb[:,:,1] * float(g) + img_rgb[:,:,2] * float(b) 
            # create JBF class
            JBF = Joint_bilateral_filter(int(sigma[1]), float(sigma[3]))
            
            bf_out = JBF.joint_bilateral_filter(img, img).astype(np.uint8)
            jbf_out = JBF.joint_bilateral_filter(img, guidance).astype(np.uint8)
        
            cv2.imwrite('gray_scale1.png', guidance)
            cv2.imwrite('filtered_rgb1.png', jbf_out)
    

            cost = np.sum(np.abs(jbf_out.astype('int32')-bf_out.astype('int32')))
        
            print(f"R: {r},G: {g},B: {b}.Cost: {cost}")
    # guidance
    # guidance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # JBF = Joint_bilateral_filter(int(sigma[1]), float(sigma[3]))
    # bf_out = JBF.joint_bilateral_filter(img, img).astype(np.uint8)
    # jbf_out = JBF.joint_bilateral_filter(img, guidance).astype(np.uint8)
    # cost = np.sum(np.abs(jbf_out.astype('int32')-bf_out.astype('int32')))
    # print(f"cv2.cvtColor.Cost: {cost}")

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    img = cv2.imread(args.image_path)
    
 
    print(args.setting_path)
    RGB, sigma = get_txt(args.setting_path)
    compute_cost(img, RGB, sigma)



if __name__ == '__main__':
    main()