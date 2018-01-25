#!/usr/bin/env sh
python ./process_boxes_visualize.py --deploy_file ./models/deploy.prototxt --model_file ./models/detect.caffemodel --input_512 ./images/t_512.jpg --input_128 ./images/t_128.jpg --gpu gpu
