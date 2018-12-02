import glob
import cv2
import numpy as np


# Parameters
base_dir = '../data/data_12'
video_name = base_dir + '/test_2_video.mov'
test_output_dir = base_dir + '/test_1_output/images/'
face_masks = base_dir + '/all_face_masks.npy'
transformation = base_dir + '/all_transformations.npy'
original_images_dir = base_dir + '/original_images/'
all_images = glob.glob(test_output_dir+ '*-outputs.png')
start = '_frame_'
end = '-outputs.png'
print(all_images[0])
all_images_sorted = sorted(all_images, key=lambda filename: int((filename.split(test_output_dir)[1].split(end)[0])))

print(original_images_dir + str(all_images_sorted[0].split(test_output_dir)[1].split(end)[0]) + '.png')
frame = cv2.imread(original_images_dir + str(all_images_sorted[0].split(test_output_dir)[1].split(end)[0]) + '.png')
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, -1, 30, (width + height * 3,height))

all_face_masks = np.load(base_dir + '/all_face_masks.npy')
all_transformations = np.load(base_dir + '/all_transformation.npy')
image_count = 0
for image in all_images_sorted:
    if image_count == 735:
        break
    if image_count % 10 == 0:
        print('On image ' + str(image_count))
    # Load the orignial frame, model output, and transformation information
    image_number = int((image.split(test_output_dir)[1].split(end)[0]))
    print(image)
    model_output = cv2.imread(image)
    original_frame = cv2.imread(original_images_dir + str(image_number) + '.png')
    original_width = all_transformations[image_number,1] - all_transformations[image_number,0]
    original_height = all_transformations[image_number,3] - all_transformations[image_number,2]

    if original_width != original_height:
        print 'PROBLEM: original width and height do not match'
    model_output = cv2.resize(model_output, (int(original_width), int(original_height)))
    l = int(all_transformations[image_number,0])
    r = int(all_transformations[image_number,1])
    t = int(all_transformations[image_number,2])
    b = int(all_transformations[image_number,3])
    
    face_mask = all_face_masks[image_number,:,:]
    center = (face_mask[4,:] + face_mask[10,:])/2
    for i in range(0, 11):
        face_mask[i,:] = face_mask[i,:] + 2.0 * (center - face_mask[i,:]) / np.linalg.norm(center - face_mask[i,:])

    # Extract the face from the model's output and paste it onto the original frame
    background_mask_face = (face_mask.reshape((-1,1,2)) - np.array([l,t])).astype(int)
    background_mask_frame = (face_mask.reshape((-1,1,2))).astype(int) 
    stencil_face = np.zeros(model_output.shape).astype(model_output.dtype)
    cv2.fillPoly(stencil_face, [background_mask_face], [1,1,1])
    model_output = model_output * stencil_face
    stencil_frame = np.ones(original_frame.shape).astype(original_frame.dtype)
    cv2.fillPoly(stencil_frame, [background_mask_frame], [0,0,0])
    original_frame = original_frame * stencil_frame
    original_frame[t:b,l:r] = original_frame[t:b,l:r] + model_output

    outline = cv2.imread(base_dir + '/test_1/' + str(image_number) + '.png')
    h, w, _ = outline.shape
    outline = cv2.resize(outline, (height * 3, height))
    print original_frame.shape
    print outline.shape
    original_frame = np.concatenate((original_frame, outline), axis=1)

    # Add the new frame to the video
    #cv2.imwrite('data_8/tmp/' + str(image_number) + '.png', original_frame)
    
    video.write(original_frame)
    image_count+=1

cv2.destroyAllWindows()
video.release()
