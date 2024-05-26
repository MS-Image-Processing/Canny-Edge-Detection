import cv2
import numpy as np
from matplotlib import pyplot as plt

def nms(magnitude, direction):
    """Performs non-maximum suppression."""

    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    
    for i in range(1, M-1):
        for j in range(1, N-1):
            if magnitude[i, j] < 10:
                continue

            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 181):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):                
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0

    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    """Performs double thresh."""
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(10)
    strong = np.int32(150)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)


def hysteresis(I, weak, strong=255):
    """Performs hysterrsis."""

    M, N = I.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (I[i, j] == weak):
                if (
                    (I[i+1, j-1] == strong) or (I[i+1, j] == strong) or
                    (I[i+1, j+1] == strong) or (I[i, j-1] == strong) or
                    (I[i, j+1] == strong) or (I[i-1, j-1] == strong) or
                    (I[i-1, j] == strong) or (I[i-1, j+1] == strong)
                ):
                    I[i, j] = strong
                else:
                    I[i, j] = 0
    return I


def show_im(i, text = "Image"):
    if type(i) == type([]):
        index = 221
        for j in range(len(i)):
            plt.subplot(index),plt.imshow(i[j], cmap=plt.get_cmap('gray')),plt.title(text[j] + ': ' + str(i[j].shape[0]) + ", " + str(i[j].shape[1]))
            plt.xticks([]), plt.yticks([])
            index = index + 1
    else: 
        plt.imshow(i, cmap=plt.get_cmap('gray'))
        plt.title(text + ': ' + str(i.shape[0]) + ", " + str(i.shape[1]) )
        plt.xticks([])
        plt.yticks([])
        plt.show()

def imread(path, i_name, ext):
    im = cv2.imread(path + i_name + "." + ext) #shapes/rect_big.png
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im
    
def Canny(img, print_imgs = 0):
    
    _img = cv2.GaussianBlur(img, (3,3), 1.5)
    
    img_x = cv2.Sobel(_img,cv2.CV_32F,1,0, 3)
    img_y = cv2.Sobel(_img,cv2.CV_32F,0,1, 3)
    
    M = np.hypot(np.abs(img_x), np.abs(img_y))
    D = np.arctan2(img_y, img_x)
    
    img_nms = nms( M, D)
    img_nms = np.clip(img_nms, 0, 255)
    
    threshold_img, weak, strong = threshold(img_nms, 0.01, 0.15)

    # print("Edges: ", weak, strong)

    # show_im(img_x, "X: ")
    # show_im(img_y, "Y: ")
    # show_im(D, "D: ")
    # show_im(M, "M: ")
    # show_im([img_nms, M], ["NMS: ", "M: "])
    # show_im(threshold_img, "Thresh: ")

    if not print_imgs == 0:
        imgs_to_print = []
        imgs_titles = []
        if "X" in print_imgs:
            imgs_to_print.append(img_x)
            imgs_titles.append("Img_X")
        if "Y" in print_imgs:
            imgs_to_print.append(img_y)
            imgs_titles.append("Img_Y")
        if "NMS" in print_imgs:
            imgs_to_print.append(img_nms)
            imgs_titles.append("NMS")
        if "Magnitude" in print_imgs:
            imgs_to_print.append(M)
            imgs_titles.append("Magnitude")
        if "Theta" in print_imgs:
            imgs_to_print.append(D)
            imgs_titles.append("Theta")
        if "Threshold" in print_imgs:
            imgs_to_print.append(threshold_img)
            imgs_titles.append("Threshold")

    final_result = hysteresis(img_nms, weak, strong)

    if print_imgs == 0:      
        return final_result
    else:
        return final_result, imgs_to_print, imgs_titles