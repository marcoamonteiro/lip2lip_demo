# lip2lip: A cGan approach to fake video generation

![](demo.gif)

This repo contains a GAN that learns how to transfer lips from a voice actor onto a source video. For more information on the process see this medium post:

Run setup_directories.sh to setup nescessary directories for training. vSave a video of your source actor in the root directory of this repo and modify 'video_path' in preprocess/face_feature_extraction_train.py to point to it.

To generate the train/test dataset run python face_feature_extraction_train.py/face_feature_extractoin_test.py from the preprocessing directory. To train the model run sh train.sh from the pix2pix-tensorflow directory. To run the model on the test set run test.sh. Finally, execute generate_video.py and add_audio.sh from the postprocessing directories to generate the new video. 

Source code for the pix2pix model came from [this](https://github.com/affinelayer/pix2pix-tensorflow) repo.

