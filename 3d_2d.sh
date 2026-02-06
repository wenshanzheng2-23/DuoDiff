export IMAGE_FOLDER="/home/yun/my_outputs/test_time/14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49/target_images"
# export IMAGE_FOLDER="/home/yun/great/vggt/test"
export OUT_IMAGE="/home/yun/my_outputs/test_time/14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49/images"

python 3d_2d.py \
    --image_folder "$IMAGE_FOLDER" \
    --out_image "$OUT_IMAGE" \
    --thr=1.0