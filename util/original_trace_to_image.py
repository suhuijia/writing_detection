# coding=utf-8
# 
import os
import sys
import cv2
import argparse
import numpy as np


def get_trace(f_file_name, save_trace_file):
    '''
    :param f_file_name: stroke file name.
    :return: trace返回公式中所有采样轨迹点的坐标值
    '''
    tf = open(save_trace_file, 'w')
    with open(f_file_name, 'r') as f:
        data = f.readlines()

    formule_x_min = 0.0
    formule_y_min = 0.0
    formule_x_max = 0.0
    formule_y_max = 0.0
    pick_flag = 0
    pick_first_cord = True

    for i, line in enumerate(data):
        if line == ".PEN_UP\n":
            pick_flag = 0
            tf.write(str(-10000) + " " + str(-10000) + "\n")

        if pick_flag == 1:
            x = line.split('\n')[0].split(" ")[0]
            y = line.split('\n')[0].split(" ")[1]
            x1 = int(round(float(x)))
            y1 = int(round(float(y)))
            tf.write(str(x1) + " " + str(y1) + "\n")

            if pick_first_cord:
                formule_x_min = float(x)
                formule_y_min = float(y)
                pick_first_cord = False
            if float(x) < formule_x_min: formule_x_min = float(x)
            if float(x) > formule_x_max: formule_x_max = float(x)
            if float(y) < formule_y_min: formule_y_min = float(y)
            if float(y) > formule_y_max: formule_y_max = float(y)

        if line == ".PEN_DOWN\n":
            pick_flag = 1
    tf.close()
    return formule_x_min, formule_x_max, formule_y_min, formule_y_max


def formule_size(trace_file_path, coord_list):
	
	xs = []
	ys = []
	if not coord_list:
		with open(trace_file_path, "r") as f:
			data = f.readlines()
		for line in data:
			x = int(line.split("\n")[0].split(" ")[0])
			y = int(line.split("\n")[0].split(" ")[1])
			if x == -10000:
				continue
			xs.append(x)
			ys.append(y)

	else:
		for list_sym in coord_list:
			for line in list_sym:
				x = int(list(line)[0])
				y = int(list(line)[1])
				xs.append(x)
				ys.append(y)

	xs_min = min(xs)
	xs_max = max(xs)
	ys_min = min(ys)
	ys_max = max(ys)
	formule_w = xs_max - xs_min
	formule_h = ys_max - ys_min

	return formule_w, formule_h, xs_min, ys_min


def draw(trace_file_path, coord_list, image_name, image_size, formule_pad, line_width):

	xs = []
	ys = []
	formule_w, formule_h, xs_min, ys_min = formule_size(trace_file_path, coord_list)
	form_size = max(formule_w, formule_h) + formule_pad
	scale = float(format(float(image_size)/float(form_size), '.5f'))
	# print(scale)
	img = np.zeros((image_size, image_size), np.uint8) * 255
	padding_w = form_size - formule_w
	padding_h = form_size - formule_h

	if not coord_list:
		with open(trace_file_path, "r") as f:
			data = f.readlines()
		for line in data:
			x = int(line.split("\n")[0].split(" ")[0])
			# x_1 = x + formule_w/2 + padding/2
			x_1 = int(round((x - xs_min + padding_w/2)*scale))
			# print(x_1)
			y = int(line.split("\n")[0].split(" ")[1])
			# y_1 = -y + formule_h/2 + padding/2
			y_1 = int(round((-1 * y - ys_min + padding_h/2)*scale))

			if x == -10000:
				for i in range(len(xs)-1):
					cv2.line(img, (xs[i], ys[i]), (xs[i+1], ys[i+1]), 255, line_width)
				xs = []
				ys = []
				continue
			xs.append(x_1)
			ys.append(y_1)

	else:
		for list_sym in coord_list:
			for line in list_sym:
				x = int(list(line)[0])
				x_1 = int(round((x - xs_min + padding_w/2)*scale))
				y = int(list(line)[1])
				y_1 = int(round((y - ys_min + padding_h/2)*scale))
				xs.append(x_1)
				ys.append(y_1)
			for i in range(len(xs)-1):
				cv2.line(img, (xs[i], ys[i]), (xs[i+1], ys[i+1]), 255, line_width)
			xs = []
			ys = []

	if not image_name:
		cv2.imwrite("./images/pad_formule.jpg", img)
	return img, padding_w, padding_h


def main(args):
	if args.f_file_name != None:
		formule_x_min, formule_x_max, formule_y_min, formule_y_max = get_trace(args.f_file_name, args.save_trace_file)
		img, padding_w, padding_h = draw(args.save_trace_file, args.coord_list, args.image_name, args.image_size, args.formule_pad, args.line_width)
	else:
		img, padding_w, padding_h = draw(args.save_trace_file, args.coord_list, args.image_name, args.image_size, args.formule_pad, args.line_width)
	
	cv2.imwrite(args.image_name, img)

	return args.image_name


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=800)
    parser.add_argument('--formule_pad', type=int, default=200)
    parser.add_argument('--line_width', type=int, default=2)
    # parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--coord_list', default=None)
    parser.add_argument('--image_name', type=str, default='./images/formula.jpg')
    parser.add_argument('--f_file_name', default=None)
                        # default="/home/nd/project/object_detect/caffe/ocr_detection/trace_file/alnumsym_21265.txt")
    parser.add_argument('--save_trace_file', default="./trace_file/001-equation000_5.trace")
    return parser.parse_args()

if __name__ == '__main__':
	main(parse_args())
	
	# f_file_name = "/home/nd/project/ND_OCR/fp/ohwfp001-160809-ef-1608091119-u2002545310far-0000.dat"
	# save_trace_file = "/home/nd/project/ND_OCR/fp_image/ohwfp001-160809-ef-1608091119-u2002545310far-0000_0.dat"
	# formule_x_min, formule_x_max, formule_y_min, formule_y_max = get_trace(f_file_name, save_trace_file)
	# image_name = "suhuijia.jpg"
	# image_size = 800
	# formule_pad = 200
	# line_width = 2
	# img, padding_w, padding_h = draw(save_trace_file, image_name, image_size, formule_pad, line_width)
	# cv2.imwrite(image_name, img)
