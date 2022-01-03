import cv2
import numpy as np
from cv2.xfeatures2d import matchGMS
from skimage.transform import resize
import argparse


MAX_FEATURES = 10000
thres_param = 3

parser = argparse.ArgumentParser()
parser.add_argument('--input1', help='Path to input image 1.', default='mf00.JPG')
parser.add_argument('--input2', help='Path to input image 2.', default='wf.JPG')
args = parser.parse_args()


def alignImages(im1, im2,MAX_FEATURES,thres_param):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    orb.setFastThreshold(0)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by score
    matches = sorted(matches, key = lambda x:x.distance)

#    # Remove not so good matches
    matches = matchGMS(im1Gray.shape[:2], im2Gray.shape[:2], keypoints1, keypoints2, matches, withScale=False, withRotation=False, thresholdFactor=thres_param)

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    outname = "Aligned_Images/" + imFilename.split(".JPG")[0] + "_aligned.jpg"
    cv2.imwrite(outname,im1Reg)
    return im1Reg 

def overlay_images(imReference,imReg,ind):
    for i in range(imReg.shape[0]):
        for j in range(imReg.shape[1]):
            if imReg[i,j].all() !=  0 :  
                imReference[center_y[ind]+i,center_x[ind]+j]= imReg[i,j]
    return imReference
if __name__ == '__main__':

    # Read reference image
    refFilename = args.input2 
    imFilename =  args.input1
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR) 


    center_x = [95,510,1015,1500,1860,2355,2950,3425,3920,4160,4650,5150]
    center_y = [1945,1990,1855,1700,1885,1850,1850,1820,1830,1775,1765,1775]
    width_offset = [1600,1600,1600,1600,1600,1600,1600,1600,1600,1600,800,800]
    height_offset = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,500,500]
if args.input1 == 'mf00.JPG':
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 0
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf01.JPG':
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 1
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf02.JPG':
    MAX_FEATURES = 25000
    thres_param = 2
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=2
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf03.JPG':
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=3
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf04.JPG':
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=4
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf05.JPG':
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=5
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf06.JPG':
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=6
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf07.JPG':
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=7
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf08.JPG':
    MAX_FEATURES = 10000
    thres_param = 3
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=8
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf09.JPG':
    MAX_FEATURES = 25000
    thres_param = 1
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=9
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'mf10.JPG':
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=10
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:12,0:width:12]
elif args.input1 == 'mf11.JPG':
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind=11
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
elif args.input1 == 'all':
    im = cv2.imread("mf00.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 0
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg1 = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    im = cv2.imread("mf01.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 1
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg2 = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    im = cv2.imread("mf02.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 2
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    MAX_FEATURES = 25000
    thres_param = 2
    imReg3 = alignImages(im, imReference_cropped,25000,2)
    im = cv2.imread("mf03.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 3
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg4 = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    im = cv2.imread("mf04.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 4
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg5 = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    im = cv2.imread("mf05.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 5
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg6 = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    im = cv2.imread("mf06.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 6
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg7 = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    im = cv2.imread("mf07.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 7
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg8 = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    im = cv2.imread("mf08.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 8
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg9 = alignImages(im, imReference_cropped,100000,3)
    im = cv2.imread("mf09.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 9
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg10 = alignImages(im, imReference_cropped,25000,1)
    im = cv2.imread("mf10.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 10
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg11 = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    im = cv2.imread("mf11.JPG", cv2.IMREAD_COLOR) 
    height = im.shape[0]
    width = im.shape[1]
    ind = 11
    imReference_cropped = imReference[center_y[ind]:center_y[ind] + height_offset[ind],center_x[ind]:center_x[ind] + width_offset[ind]]
    im = im[0:height:6,0:width:6]
    imReg12 = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    

if args.input1 != 'all':
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    imReg = alignImages(im, imReference_cropped,MAX_FEATURES,thres_param)
    res_ = overlay_images(imReference,imReg,ind)
    # Write aligned image to disk. 
    outFilename = "Results/"+ imFilename.split(".JPG")[0]+ "_result.jpg" 
    resized_image = cv2.resize(res_,(1200,800))
    cv2.imwrite(outFilename, res_) 
    cv2.imshow("Result",resized_image)
    cv2.waitKey()
else:
    res_ = overlay_images(imReference,imReg1,0)
    res_ = overlay_images(res_,imReg2,1)
    res_ = overlay_images(res_,imReg3,2)
    res_ = overlay_images(res_,imReg4,3)
    res_ = overlay_images(res_,imReg5,4)
    res_ = overlay_images(res_,imReg6,5)
    res_ = overlay_images(res_,imReg7,6)
    res_ = overlay_images(res_,imReg8,7)
    res_ = overlay_images(res_,imReg9,8)
    res_ = overlay_images(res_,imReg10,9)
    res_ = overlay_images(res_,imReg11,10)
    res_ = overlay_images(res_,imReg12,11)
    resized_image = cv2.resize(res_,(1200,800))
    outFilename = "Results/" + "all_results.jpg"
    cv2.imwrite(outFilename, res_) 
    cv2.imshow("Result",resized_image)

