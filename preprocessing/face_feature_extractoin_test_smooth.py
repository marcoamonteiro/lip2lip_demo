import numpy as np
import cv2
import dlib
import math
import sys
import pickle
import argparse
import os
import skvideo.io
import random
import itertools

def get_ellipse_parameters(l, t, r, b):
    center = np.array([1.0*(l[0]+r[0])/2, 1.0*(l[1]+r[1])/2])
    major_axis_vector = np.array([1.0*r[0]-l[0], 1.0*r[1]-l[1]])
    minor_axis_vector = np.array([1.0*t[0]-b[0], 1.0*t[1]-b[1]])
    major_axis_length = int(np.sqrt(np.dot(major_axis_vector, major_axis_vector))/1.4 + np.random.normal(0, 0.13,1))
    minor_axis_length = int(np.sqrt(np.dot(minor_axis_vector, minor_axis_vector))/1.1 + np.random.normal(0, 0.13,1))
    center = center - minor_axis_vector / 4
    e_x = np.array([1,0])
    angle =  -180 / np.pi *np.arccos(np.dot(major_axis_vector, e_x)/(np.sqrt(np.dot(major_axis_vector, major_axis_vector)))) #+ np.random.normal(0, 7, 1)
    if l[1] < r[1]:
        angle *= -1
    return (int(center[0]), int(center[1])), (major_axis_length, minor_axis_length), angle

def get_crop_bounds(face, features):
    leftmost_face_feature = float('Inf')
    rightmost_face_feature = -1
    lowermost_face_feature = -1
    topmost_face_feature = float('Inf')
    for ii in range(1, 18):
        feature = features.part(ii)
        leftmost_face_feature = min(leftmost_face_feature, feature.x)
        rightmost_face_feature = max(rightmost_face_feature, feature.x)
        lowermost_face_feature = max(lowermost_face_feature, feature.y)
        topmost_face_feature = min(topmost_face_feature, feature.y)

    width = rightmost_face_feature - leftmost_face_feature
    height = lowermost_face_feature - topmost_face_feature
    if width < height:
        leftmost_face_feature -= (height - width) / 2
        rightmost_face_feature = leftmost_face_feature + height
    elif width > height:
        topmost_face_feature -= (width - height) / 2
        lowermost_face_feature = topmost_face_feature + width
    return leftmost_face_feature, rightmost_face_feature, topmost_face_feature, lowermost_face_feature

def blackout_background(image, face_features, test_image_number, all_face_masks):
    background_mask = np.zeros((15,2))
    for i in range(1,16):
        jaw_point = np.array([face_features[i][0], face_features[i][1]])
        background_mask[i-1,:] = jaw_point
    all_face_masks[test_image_number,:,:] = background_mask
    background_mask = (background_mask.reshape((-1,1,2))).astype(int)
    stencil = np.zeros(image.shape).astype(image.dtype)
    cv2.fillPoly(stencil, [background_mask], [255,255,255])
    result = cv2.bitwise_and(image, stencil)
    return result

def blackout_jaw(image, face_features, make_stencil=False, l=0, t=0, r=0, b=0, path=''):
    face_mask = np.zeros((12,2))
    mouth_center = np.array([face_features[62][0], face_features[62][1]])
    for i in range(4, 13):
        jaw_point = np.array([face_features[i][0], face_features[i][1]])
        vec = mouth_center-jaw_point
        vec = vec/np.linalg.norm(vec)*10
        jaw_point = jaw_point+vec 
        face_mask[i-4,:] = jaw_point
    under_nose_point = np.array([face_features[33][0], face_features[33][1]])
    vec = mouth_center-under_nose_point
    vec = (vec/np.linalg.norm(vec)*5).astype(int)
    right_top_point = np.array([face_features[13][0], face_features[13][1]]) 
    left_top_point = np.array([face_features[3][0], face_features[3][1]]) 
    face_mask[9,:] = (under_nose_point+right_top_point)/2
    face_mask[11,:] = (under_nose_point+left_top_point)/2
    under_nose_point+=vec 
    face_mask[10,:] = under_nose_point
      
    face_mask = (face_mask.reshape((-1,1,2))).astype(int)
    cv2.fillPoly(image,[face_mask],(255,0,255))

    if make_stencil:
        stencil = np.zeros(image.shape).astype(image.dtype)
        cv2.fillPoly(stencil, [face_mask], (255,255,255))
        stencil = stencil[t:b, l:r]
        stencil = cv2.resize(stencil, (256,256))
        cv2.imwrite(path, stencil)



def make_blackout_jaw_stencil(face_features):
    stencil = np.zeros(image.shape).astype(image.dtype)
    cv2.fillPoly(stencil, [face_mask], (255,255,255))
    return cv2.bitwise_and(image, stencil)

def draw_lips(image, dst_lip_features, original_lip_features):
    # Translate
    lip_offset = original_lip_features[14,:] - dst_lip_features[14,:]
    dst_lip_features = (dst_lip_features + lip_offset)

    # Rotate and Scale
    
    dst_axis = dst_lip_features[6,:] - dst_lip_features[0,:]
    original_axis = original_lip_features[6,:] - original_lip_features[0,:]
    cosine = np.dot(dst_axis, original_axis)/np.sqrt(np.dot(dst_axis, dst_axis))/np.sqrt(np.dot(original_axis, original_axis))
    cosine = max(min(cosine, 1.0), -1.0) # Floating point error
    angle = np.arccos(cosine)
    if dst_axis[1] > original_axis[1]:
        angle = angle*-1
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]]) 
    mouth_center = (dst_lip_features[14,:] + dst_lip_features[18,:])/2
    scale = np.linalg.norm(original_axis) / np.linalg.norm(dst_axis)

    dst_lip_features = dst_lip_features - mouth_center
    dst_lip_features = np.matmul(rotation_matrix, dst_lip_features.T)
    dst_lip_features = dst_lip_features * scale
    dst_lip_features = dst_lip_features.T + mouth_center
    

    #Scale
    #scale = np.linalg.norm(original_axis) / np.linalg.norm(dst_axis)
    #dst_lip_features = dst_lip_features * scale

    dst_lip_features = dst_lip_features.astype(int)

    cv2.fillPoly(image,[dst_lip_features[np.r_[0:7,16,15,14,13,12]]],(255,0,0))
    cv2.fillPoly(image,[dst_lip_features[np.r_[12:20]]],(0,255,0))
    cv2.fillPoly(image,[dst_lip_features[np.r_[6:12,0, 12, 19,18,17,16]]],(0,0,255))

# Parameters
original_video = 'video2.mp4'
lips_video = 'video3.mp4'
base_data_dir = 'data_8/'
test_dir_1 = base_data_dir + 'test_1'
test_dir_2 = base_data_dir + 'test_3'
tmp_dir = base_data_dir + 'tmp'
stencil_path = base_data_dir + 'jaw_stencil'
original_images_dir = base_data_dir + 'original_images'
num_test_images = 740
all_lip_features = np.zeros((num_test_images, 20, 2))
all_face_masks = np.zeros((num_test_images, 15, 2))
all_transformations = np.zeros((num_test_images, 4))
# Load face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load video
original_video_reader = skvideo.io.FFmpegReader(original_video)
lips_video_reader = skvideo.io.FFmpegReader(lips_video)
video_shape = original_video_reader.getShape()
(num_frames, h, w, c) = video_shape
print('Num frames ' + str(num_frames))
frame_count = 0
prev_image = None
prev_outline = None

for frame, lips_frame in itertools.izip(original_video_reader.nextFrame(), lips_video_reader.nextFrame()):
    print frame_count
    if frame_count >= num_test_images:
        break
    if frame_count % 100 == 0:
        print('On frame ' + str(frame_count) + ' of ' + str(num_frames))
    face_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    face_image_lips = cv2.cvtColor(lips_frame, cv2.COLOR_RGB2BGR)
    original_image = np.copy(face_image)
    faces_original = detector(frame, 1)
    faces_lips = detector(lips_frame, 1)

    if len(faces_original) > 1 or len(faces_lips) > 1:
        print('DETECTED MORE THAN ONE FACE')
        continue
    if len(faces_original) == 0 or len(faces_lips) == 0:
        print('DETECTED NO FACES')
        continue

    # Extract face features
    principal_face = faces_original[0]
    shape = predictor(frame, principal_face)
    principal_face_lips = faces_lips[0]
    shape_lips = predictor(lips_frame, principal_face_lips)

    # Get Crop Bounds
    leftmost_face_feature, rightmost_face_feature, topmost_face_feature, lowermost_face_feature = get_crop_bounds(principal_face, shape)
    all_transformations[frame_count, :] = np.array([leftmost_face_feature, rightmost_face_feature, topmost_face_feature, lowermost_face_feature]) 
    face_image_annotated = np.copy(face_image)

    # Save face and lip features in array
    lip_features_original = np.zeros((20,2))
    lip_features_src = np.zeros((20,2))
    face_features = []
    co = 0
    for ii in range(0, 68):
        X = shape.part(ii)
        X_lips = shape_lips.part(ii)
        face_features.append((X.x, X.y))
        co += 1
        if ii>=48:
            lip_features_src[ii-48] = np.array([X_lips.x - leftmost_face_feature, X_lips.y - topmost_face_feature])
            lip_features_original[ii-48,:] = np.array([X.x - leftmost_face_feature, X.y-topmost_face_feature])
    lip_features_original = lip_features_original.astype(int)

    # Blackout jaw and background
    blackout_jaw(face_image_annotated, face_features, True, leftmost_face_feature, topmost_face_feature, rightmost_face_feature, lowermost_face_feature, stencil_path + '/' + str(frame_count) + '.png')
    face_image_annotated = blackout_background(face_image_annotated, face_features, frame_count, all_face_masks)
    face_image = blackout_background(face_image, face_features, frame_count, all_face_masks)

    face_image = face_image[topmost_face_feature:lowermost_face_feature,leftmost_face_feature:rightmost_face_feature]
    face_image_annotated = face_image_annotated[topmost_face_feature:lowermost_face_feature,leftmost_face_feature:rightmost_face_feature]
    
    # Draw Lips
    draw_lips(face_image_annotated, lip_features_src, lip_features_original)
    
    # Scale and concatenate
    original_image_size = face_image.shape[0]
    face_image = cv2.resize(face_image, (256,256))
    face_image_annotated = cv2.resize(face_image_annotated, (256,256))
    face_image_lips = cv2.resize(face_image_lips, (256,256))
    if frame_count < 0:
        #print face_image.shape, face_image_annotated.shape, prev_image.shape, prev_outline.shape
        #test_image = np.concatenate((face_image, face_image_annotated, prev_image, prev_outline), axis = 1)
        
        # Stack and save images. Change this for pix2pixHD
        # all_lip_features[frame_count,:,:] = 1.0 * lip_features / original_image_size * 256
        cv2.imwrite(tmp_dir+ '/_original_' + str(frame_count).zfill(6) + '.png', face_image)
        # cv2.imwrite(test_dir_1 + '/_frame_' + str(frame_count).zfill(6) + '.png', test_image)
        #cv2.imwrite(tmp_dir+ '/_annotated_' + str(frame_count) + '.png', face_image_annotated)
        cv2.imwrite(original_images_dir + '/' + str(frame_count).zfill(6) + '.png', original_image)
        cv2.imwrite(test_dir_1 + '/' + str(frame_count) + '.png', face_image_annotated)
        '''
        test_image[:,:256,:]=0
        test_image[:,768:,:]=0
        if frame_count > 1:
            test_image[:,512:768,:]=0
        cv2.imwrite(test_dir_1 + '/_frame_' + str(frame_count) + '.png', test_image)
        '''
    if True:
        cv2.imwrite(test_dir_1 + '/bootstrap.png', face_image)
    prev_image = face_image 
    prev_outline = face_image_annotated

    frame_count += 1

'''
for i in range(0, num_test_images/2):
    if i % 100 == 0:
        print 'On iteration ' + str(i) + ' of ' + str(num_test_images/2)
    image1 = cv2.imread(tmp_dir+'/_original_'+str(i)+'.png', cv2.IMREAD_COLOR)
    image1_annotated = cv2.imread(tmp_dir+'/_annotated_'+str(i)+'.png', cv2.IMREAD_COLOR)
    image1_lips = all_lip_features[i,:,:]
    image2 = cv2.imread(tmp_dir+'/_original_'+str(int(i + num_test_images/2))+'.png', cv2.IMREAD_COLOR)
    image2_annotated = cv2.imread(tmp_dir+'/_annotated_'+str(int(i + num_test_images/2))+'.png', cv2.IMREAD_COLOR)
    image2_lips = all_lip_features[int(i + num_test_images/2),:,:]

    draw_lips(image1_annotated, image2_lips, image1_lips)
    draw_lips(image2_annotated, image1_lips, image2_lips)
    image1_concat = np.concatenate((image2, image1_annotated, image1), axis=1)
    image2_concat = np.concatenate((image1, image2_annotated, image2), axis=1)
    cv2.imwrite(test_dir_1 + '/_frame_' + str(i) + '.png', image1_concat)
    cv2.imwrite(test_dir_2 + '/_frame_' + str(i + num_test_images/2) + '.png', image2_concat)
    cv2.imwrite('data_5/test_output/_frame_' + str(i) + '-outputs.png', image1_annotated)
    cv2.imwrite('data_5/test_output/_frame_' + str(i + num_test_images/2) + '-outputs.png', image2_annotated)
'''

#all_face_masks = all_face_masks[1,:,:]
#all_transformations = all_transformations[1:,:]
np.save(base_data_dir + 'all_face_masks', all_face_masks)
np.save(base_data_dir + 'all_transformation', all_transformations)
# Script 1: Test video to images in folder
# Script 2: Images in folder to blacked out faces ready for the model. 
# Also save a dict with sufficient info to paste faces back on image
# Script 3: Paste faces from model result back on original image and combine images to make a video
