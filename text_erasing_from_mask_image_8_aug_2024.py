import numpy as np
import cv2


def text_eraser_from_mask_images(source_image,mask_image):
    
    image_Gray = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
    height, width = image_Gray.shape[:2]
    # Create a blank (black) image
    blank_image = np.zeros((height, width), dtype=np.uint8)
    # cv2.imwrite("blank_image.jpg",blank_image)
    _, thresh = cv2.threshold(image_Gray, 40, 255, cv2.THRESH_BINARY & cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    cv2.imwrite("ca_dana_point_thresh.jpg",thresh)
    

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    print("Original contours length",len(contours))
    retained_contours = []
    for i,cont in enumerate(contours):
   
        image_with_text = blank_image.copy()
        cnt = np.array([cont], np.int32)
        cnt = cnt.reshape((-1, 1, 2))

        cv2.fillPoly(image_with_text, [cnt], (255,255,255))

        filled_area = cv2.countNonZero(image_with_text)
        # total image area
        total_image_area = image_Gray.shape[0]*image_Gray.shape[1]
        if filled_area <= 0.01*total_image_area:

            result_image = cv2.bitwise_and(image_with_text, mask_image) # Perform logical AND operation with the source mask image
            
            if np.any(result_image):  # Check if the result image is blank
                retained_contours.append(cont)
    return retained_contours

intersected_contours = []

# def retain_intersected_contours(retained_contours,source_mask_image):
#     height,width = source_mask_image.shape[:2]
#     # print("Source Mask Image shape",source_mask_image)
#     blank_image = np.zeros((height,width),dtype=np.uint8)
        
    
#     for i,cnt211 in enumerate(retained_contours):
#         # print(cnt211)
#         blank_image_with_text = blank_image.copy()
#         cnt11 = np.array([cnt211], np.int32)
#         cnt11 = cnt11.reshape((-1, 1, 2))

#         cv2.fillPoly(mask_image, [cnt11], (255))
    
#         intersections = cv2.bitwise_and(blank_image_with_text,source_mask_image)
#         intersection_area = np.sum(intersections)
#         bbox_mask_intersection_area = np.sum(blank_image_with_text)
#         # print("bbox_mask_intersection_area",bbox_mask_intersection_area)
#         if bbox_mask_intersection_area==0:
#             return 0
#         intersection_percentage = intersection_area/bbox_mask_intersection_area
#         # print("intersection percentage",intersection_percentage)
#         if intersection_percentage >=0.01:
#             intersected_contours.append(cnt211)
#             # print("appended")

#     return intersected_contours

source_image_path = "/home/usama/Converted_jpg_from_tiff_july3_2024/ca_colma.jpg"
mask_image_path = "/home/usama/Denoised_mask_results_3_july_2024/ca_colma/ca_colma_1_mask.jpg"
source_image = cv2.imread(source_image_path)
mask_image = cv2.imread(mask_image_path)
mask_image = cv2.cvtColor(mask_image,cv2.COLOR_BGR2GRAY)
retained_contours = text_eraser_from_mask_images(source_image,mask_image)


print("Length of retained Contours",len(retained_contours))
# print("length of intersected Contours",len(intersected_contours))

height,width = mask_image.shape[:2]
blank_mask_image = np.zeros((height,width),dtype = np.uint8)
kernel = np.ones((3, 3), np.uint8)
       
for i,cnt1 in enumerate(retained_contours):
    cnt11 = np.array([cnt1], np.int32)
    cnt11 = cnt11.reshape((-1, 1, 2))

    cv2.fillPoly(blank_mask_image, [cnt11], (255))
    # cv2.dilate(blank_mask_image,kernel,iterations=1)
    cv2.fillPoly(mask_image, [cnt11], (255))
    # merged_ca_dublin_result = cv2.bitwise_or(blank_mask_image,mask_image)
    # cv2.drawContours(blank_mask_image,[cnt1],-1,(255,255,255),4)
    # cv2.imwrite(f"/home/usama/Aug_1_2024_code/retained_contours/retained_contours{i}.jpg",blank_mask_image)
kernel = np.ones((3, 3), np.uint8)

cv2.imwrite("text_area_masks_ca_colma_1.jpg",blank_mask_image)
mask_image_dilated = cv2.dilate(blank_mask_image,kernel,iterations=1)
merged_result = cv2.bitwise_or(mask_image_dilated,mask_image)
cv2.imwrite("removed_text_ca_colma_1.jpg",mask_image)
cv2.imwrite("removed_merged_text_ca_colma_1.jpg",merged_result)

# intersected_contours.clear()
