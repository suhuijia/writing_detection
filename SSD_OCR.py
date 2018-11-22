# coding=utf-8
# 
import copy
import os
import sys
import cv2
import argparse
from util import ocr_detect
from util import original_trace_to_image

curDir = os.getcwd()
# print(curDir)
sys.path.insert(0, curDir)


class SSD_OCR():
	"""docstring for SSD_OCR"""
	def __init__(self, model_weights, image_size=800, formule_pad=200):
		self.model_weights = model_weights
		self.image_size = image_size
		self.formule_pad = formule_pad

	def test_ocr(self, trace_seq):
		'''
		trace_seq: [300, 200, 300, 400, -10000, -10000, 800, 200, 800, 400, -10000, -10000, ...]
		'''
		traces = self.process_trace(trace_seq)
		save_img_args = original_trace_to_image.parse_args()
		if traces != None:
			save_img_args.coord_list = traces

		det_img_args = ocr_detect.parse_args()
		# save_img_args, det_img_args = self.main(save_img_args, det_img_args, save_trace_file)
		save_img_args, det_img_args = self.main(save_img_args, det_img_args)
		# print(save_img_args)
		# print(save_img_args.image_name)
		img = original_trace_to_image.main(save_img_args)
		img = cv2.imread(img)
		cv2.imshow("save_img",img)
		cv2.waitKey(1000)
		result_box_all, result_sym_top5, result_conf_top5 = ocr_detect.main(det_img_args)
		return result_box_all, result_sym_top5, result_conf_top5

	def main(self, save_img_args, det_img_args):

		if self.image_size != None:
			save_img_args.image_size = self.image_size
		if self.formule_pad != None:
			save_img_args.formule_pad = self.formule_pad
		# if save_trace_file != None:
		# 	save_img_args.save_trace_file = save_trace_file
		if self.model_weights != None:
			det_img_args.model_weights = self.model_weights
		if det_img_args.image_file != save_img_args.image_name:
			det_img_args.image_file = save_img_args.image_name
		return save_img_args, det_img_args

	def process_trace(self, trace_seq):
	    traces = []
	    stroke = []
	    for i in range(1, len(trace_seq), 2):
	        x = trace_seq[i-1]
	        y = trace_seq[i]
	        if x != -10000:
	            stroke.append([x, y])
	        else:
	            traces.append(copy.copy(stroke))
	            stroke = []

	    return traces


if __name__ == '__main__':

	save_trace_file = "./trace_file/001-equation000_5.trace"
	model_weights = "model/VGG_three_grade_SSD_300x300_iter_85000.caffemodel"
	# image_name = "./images/pad_formula(3).jpg"   # 保存的图片
	# image_file = "./images/formula.jpg"   # 检测时读取的图片

	pt_seq = [300, 200, 300, 400, -10000, -10000, 800, 200, 800, 400, -10000, -10000]
	model = SSD_OCR(model_weights)
	result_box_all, result_sym_top5, result_conf_top5 = model.test_ocr(pt_seq)

	print "top5 of result: "
	print(result_box_all)
	print(result_sym_top5)
	print(result_conf_top5)
