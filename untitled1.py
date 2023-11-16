import numpy as np
import cv2
import matplotlib.pyplot as plt

# Feature Detection and Description
def detect_and_describe(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Feature Matching
def match_keypoints(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match des1 and des2
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    return good_matches

# Homography Estimation
def estimate_homography(kp1, kp2, des1, des2):
    # Match keypoints and descriptors between des1 and des2
    good_matches = match_keypoints(des1, des2)
    
    if not good_matches:
        print("No valid matches found between the images.")
        return None

    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Use RANSAC to find the homography matrix
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return homography

# Image Warping and Blending
def warp_and_blend_images(img1, img2, homography):
    # Warp the second image to align with the first image
    result = cv2.warpPerspective(img1, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    return result

# Read images
img3 = cv2.imread('image3.jpg')
img2 = cv2.imread('image4.jpg')
img1 = cv2.imread('image5.jpg')

# Feature Detection and Description
kp1, des1 = detect_and_describe(img1)
kp2, des2 = detect_and_describe(img2)
kp3, des3 = detect_and_describe(img3)

# Feature Matching and Blending for Image 1 and Image 2
good_matches1_2 = match_keypoints(des1, des2)
homography1_2 = estimate_homography(kp1, kp2, des1, des2)
result1_2 = warp_and_blend_images(img1, img2, homography1_2)

# Feature Matching and Blending for Result of Image 1 and 2 and Image 3
kp_result, des_result = detect_and_describe(result1_2)
good_matches_result_3 = match_keypoints(des_result, des3)
homography_result_3 = estimate_homography(kp_result, kp3, des_result, des3)
final_result = warp_and_blend_images(result1_2, img3, homography_result_3)

# Display the final result
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.show()

