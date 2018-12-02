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

def get_crop_bounds(face, features):
    leftmost_face_feature = float('Inf')
    rightmost_face_feature = -1
    lowermost_face_feature = -1
    topmost_face_feature = float('Inf')
    for i in range(1, 18):
        feature = features.part(i)
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
    background_mask = np.zeros((13,2))
    for i in range(2,15):
        jaw_point = np.array([face_features[i][0], face_features[i][1]])
        background_mask[i-2,:] = jaw_point
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

def make_lip_image(lip_features):
    # Rotate and Scale
    original_axis = np.array([1.0,0.0])
    dst_axis = lip_features[6,:] - lip_features[0,:]
    cosine = np.dot(dst_axis, original_axis)/np.sqrt(np.dot(dst_axis, dst_axis))/np.sqrt(np.dot(original_axis, original_axis))
    cosine = max(min(cosine, 1.0), -1.0) # Floating point error
    angle = np.arccos(cosine)
    if dst_axis[1] > original_axis[1]:
        angle = angle*-1
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]]) 
    mouth_center = (lip_features[14,:] + lip_features[18,:])/2

    lip_features = lip_features - mouth_center
    lip_features = np.matmul(rotation_matrix, lip_features.T)
    lip_features = lip_features.T + mouth_center

    left = float('inf')
    top = float('inf')
    right = 0
    bottom = 0

    for i in range(lip_features.shape[0]):
        left = min(left, lip_features[i, 0])
        top = min(top, lip_features[i, 1])
        right = max(right, lip_features[i,0])
        bottom = max(bottom, lip_features[i,1])

    translate = np.array([left, top]).T
    lip_features = lip_features - translate

    lip_width = int(right - left)
    lip_height = int(bottom - top)

    if lip_width > lip_height:
        offset = (lip_width - lip_height) / 2
        translate = np.array([0, offset]).T
        lip_features = lip_features + translate 
    else:
        offset = (lip_height - lip_width) / 2
        translate = np.array([offset, 0]).T
        lip_features = lip_features + translate  

    lip_features = lip_features.astype(int)

    lip_outline_image = np.zeros((max(lip_width, lip_height), max(lip_width, lip_height), 3))
    cv2.fillPoly(lip_outline_image,[lip_features[np.r_[0:7,16,15,14,13,12]]],(255,0,0))
    cv2.fillPoly(lip_outline_image,[lip_features[np.r_[12:20]]],(0,255,0))
    cv2.fillPoly(lip_outline_image,[lip_features[np.r_[6:12,0, 12, 19,18,17,16]]],(0,0,255))
    return lip_outline_image

# Parameters
original_video = '../obama.mp4'
lips_video = '../clip2.mov'
base_data_dir = '../data/data_10/'
test_dir_1 = base_data_dir + 'test_1'
stencil_path = base_data_dir + 'jaw_stencil'
original_images_dir = base_data_dir + 'original_images'
num_test_images = 150

# Load face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load video
original_video_reader = skvideo.io.FFmpegReader(original_video)
lips_video_reader = skvideo.io.FFmpegReader(lips_video)
video_shape = original_video_reader.getShape()
(num_frames, h, w, c) = video_shape
(num_test_images, _, _, _) = lips_video_reader.getShape()
print('Number of frame: ' + str(num_test_images))
all_face_masks = np.zeros((num_test_images, 13, 2))
all_transformations = np.zeros((num_test_images, 4))
frame_count = 0

first = True
orig_frame = None
for frame, lips_frame in itertools.izip(original_video_reader.nextFrame(), lips_video_reader.nextFrame()):
    if first == True:
        first = False
        orig_frame = frame
    else:
        frame = orig_frame
    if frame_count >= num_test_images:
        break
    if frame_count % 10 == 0:
        print('On frame ' + str(frame_count) + ' of ' + str(num_frames))
    face_image_annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    face_image_lips = cv2.cvtColor(lips_frame, cv2.COLOR_RGB2BGR)
    original_image = np.copy(face_image_annotated)

    original_faces = detector(frame, 1)
    actor_faces = detector(lips_frame, 1)

    if len(original_faces) > 1 or len(actor_faces) > 1:
        print('DETECTED MORE THAN ONE FACE')
        continue
    if len(original_faces) == 0 or len(actor_faces) == 0:
        print('DETECTED NO FACES')
        continue

    # Extract face features
    principal_face_original = original_faces[0]
    shape = predictor(frame, principal_face_original)
    principal_face_lips = actor_faces[0]
    shape_lips = predictor(lips_frame, principal_face_lips)

    # Get Crop Bounds
    leftmost_face_feature, rightmost_face_feature, topmost_face_feature, lowermost_face_feature = get_crop_bounds(principal_face_original, shape)
    all_transformations[frame_count, :] = np.array([leftmost_face_feature, rightmost_face_feature, topmost_face_feature, lowermost_face_feature]) 

    # Save face and lip features in array
    lip_features_actor = np.zeros((20,2))
    face_features_original = []
    for i in range(0, 68):
        X = shape.part(i)
        X_lips = shape_lips.part(i)
        face_features_original.append((X.x, X.y))
        if i>=48:
            lip_features_actor[i-48] = np.array([X_lips.x, X_lips.y])

    # Blackout jaw and background
    blackout_jaw(face_image_annotated, face_features_original, True, leftmost_face_feature, topmost_face_feature, rightmost_face_feature, lowermost_face_feature, stencil_path + '/' + str(frame_count) + '.png')
    face_image_annotated = blackout_background(face_image_annotated, face_features_original, frame_count, all_face_masks)
    face_image_annotated = face_image_annotated[topmost_face_feature:lowermost_face_feature,leftmost_face_feature:rightmost_face_feature]
    
    # Draw Lips
    lips_outline_image = make_lip_image(lip_features_actor)

    # Scale and concatenate
    face_image_lips = cv2.resize(face_image_lips, (256,256))
    face_image_annotated = cv2.resize(face_image_annotated, (256,256))
    lips_outline_image = cv2.resize(lips_outline_image, (256,256))
    test_image = np.concatenate((face_image_lips, face_image_annotated, lips_outline_image), axis = 1)
    
    cv2.imwrite(original_images_dir + '/' + str(frame_count) + '.png', original_image)
    cv2.imwrite(test_dir_1 + '/' + str(frame_count) + '.png', test_image)

    frame_count += 1
    
np.save(base_data_dir + 'all_face_masks', all_face_masks)
np.save(base_data_dir + 'all_transformation', all_transformations)