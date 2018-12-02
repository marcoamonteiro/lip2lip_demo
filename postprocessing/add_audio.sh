#!/bin/bash 
VIDEO_PATH=../data/data_10/test_4_video.mov
VIDEO_WITH_AUDIO_PATH=../clip2.mov
AUDIO_SAVE_PATH=../data/data_10/output-audio-2.aac
OUTPUT_VIDEO_PATH=../data/data_10/output-video-2.mov

ffmpeg -i $VIDEO_WITH_AUDIO_PATH -vn -acodec copy $AUDIO_SAVE_PATH
ffmpeg -i $VIDEO_PATH -i $AUDIO_SAVE_PATH -codec copy -shortest $OUTPUT_VIDEO_PATH