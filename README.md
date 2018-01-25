# SingleShortSemanticMatcher

Caffe Implementation of Semantic Matcher

## Model files
./models contains caffemodel and prototxt files with model:
- deploy.prototxt
- detect.caffemodel (*input_image: 512x512x3*)

## Main file usage
run 
```
./vis_res.sh 
```
OR 
```
python ./process_boxes_visualize.py --deploy_file ./models/deploy.prototxt --model_file ./models/detect.caffemodel --input_512 ./images/t_512.jpg --input_128 ./images/t_128.jpg --gpu gpu
```
## Requirements
Caffe, pycaffe

## Example:
<p align="left">  <img src="https://github.com/NIIAS3050/SingleShortSemanticMatcher/blob/master/images/example.png"></p>
