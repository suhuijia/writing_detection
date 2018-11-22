# coding=utf-8
# 

import os
import sys
import cv2
import argparse
from util import ocr_detect
from util import original_trace_to_image

curDir = os.getcwd()
print(curDir)
sys.path.insert(0, curDir)


def test_ocr(args):
	'''
	save_img_args, det_img_args all args
	'''
	save_img_args = original_trace_to_image.parse_args()
	det_img_args = ocr_detect.parse_args()
	save_img_args, det_img_args = main(args, save_img_args, det_img_args)

	print(save_img_args)
	print(save_img_args.image_name)
	img = original_trace_to_image.main(save_img_args)
	img = cv2.imread(img)
	cv2.imshow("save_img",img)
	cv2.waitKey(2000)

	ocr_detect.main(det_img_args)


def main(args, save_img_args, det_img_args):

	if args.image_size != None:
		save_img_args.image_size = args.image_size
	if args.formule_pad != None:
		save_img_args.formule_pad = args.formule_pad
	if args.image_name != None:
		save_img_args.image_name = args.image_name
	if args.save_trace_file != None:
		save_img_args.save_trace_file = args.save_trace_file
	if args.model_weights != None:
		det_img_args.model_weights = args.model_weights
	if args.image_file != None:
		det_img_args.image_file = args.image_file

	if det_img_args.image_file != save_img_args.image_name:
		det_img_args.image_file = save_img_args.image_name

	return save_img_args, det_img_args


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=800)
    parser.add_argument('--formule_pad', type=int, default=200)
    parser.add_argument('--line_width', type=int, default=2)
    parser.add_argument('--coord_list', default=None)
    parser.add_argument('--image_name', type=str, default='./images/formula123.jpg')
    # parser.add_argument('--f_file_name', default=None)
                        # default="/home/nd/project/object_detect/caffe/ocr_detection/trace_file/alnumsym_21265.txt")
    parser.add_argument('--save_trace_file', default="./trace_file/alnumsym_21273.txt")
    parser.add_argument('--model_weights',
                        default='model/VGG_three_grade_SSD_300x300_iter_80000.caffemodel')
    parser.add_argument('--image_file', default='images/formula.jpg')
    return parser.parse_args()


if __name__ == '__main__':
	test_ocr(parse_args())
