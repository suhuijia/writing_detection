#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2
import time

# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        # caffe.set_device(gpu_id)
        caffe.set_mode_cpu()
        # if self.gpu_id is None:
        #     caffe.set_mode_cpu()
        # else:
        #     caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([128, 128, 128])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)


    def detect(self, image_file, conf_thresh=0.45, topn=60):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        start = time.time()
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        prob= self.net.blobs['mbox_conf_softmax'].data[0]

        # Forward pass.
        detections = self.net.forward()['detection_out']
        # list = detections[0, :, i, 0]
        # print(len(detections[0]), len(detections[0][0]), len(detections[0][0][0]))

        # print(detections)
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        # print(det_label)
        det_conf = detections[0,0,:,2]
        # print(det_conf)
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
        # print(top_indices)
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        print(len(top_conf))
        result = []
        det_top_5_label = []
        det_top_5_conf = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
            print(result)
            # top_5 result
            # prob= self.net.blobs['mbox_conf_softmax'].data[0] #.flatten()
            # print(type(prob))
            # print(prob[0][0])
            # print(np.max(prob[0]))
            # order=prob.argsort()[-1]
            list = []
            result_conf = []
            order = np.argsort(prob)
            for i in range(len(prob)):
            	sort = order[i]
            	# print(sort) 
            	if sort[-1] == int(label):
            		list.append(i)
            		conf = prob[i][int(label)]; #print(conf)
            		if round(float(conf), 10) == round(float(score), 10):
            			result_conf.append(i)
            			
            if len(result_conf) == 0:
            	for i in range(len(prob)):
            		sort = order[i]
            		if sort[-2] == int(label):
            			conf = prob[i][int(label)]
            			if round(float(conf), 10) == round(float(score), 10):
            				result_conf.append(i)
            # print(list)
            # print(len(list))
            
            print(result_conf)
            print(len(result_conf))
            top_5_label = order[result_conf[0]][-1:-6:-1].tolist()
            # print(type(top_5_label[0]))
            if int(0) in top_5_label:
            	top_5_label.remove(0)
            	top_5_label.append(order[result_conf[0]][-6])
            # print "top_5_label: ", top_5_label

            top_5_conf = []
            for j in top_5_label:
            	top_5_conf.append(prob[result_conf[0]][j]) #   [i for i in top_5_label]
            # print "top_5_conf: ", top_5_conf
            det_top_5_label.append(top_5_label)
            det_top_5_conf.append(top_5_conf)

        time1 = time.time() - start
        print(time1)
        return result, det_top_5_label, det_top_5_conf


def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    start = time.time()
    result, det_top_5_label, det_top_5_conf = detection.detect(args.image_file)
    # time1 = time.time() - start
    # print "Detect:", result

    img = Image.open(args.image_file)
    print "Image name:", args.image_file
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    width, height = img.size
    print "Image size:", width, height
    syms = read_sym_name(args.map_file)
    result_box_all = []
    result_sym_top5 = []
    result_conf_top5 = []
    for i, item in enumerate(result):
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        draw.rectangle([xmin, ymin, xmax, ymax], fill=None, outline=(255, 0, 0))
        # draw.rectangle([xmin, ymin, xmax, ymax], outline=(255))
        # draw.text([xmin, ymin], item[-1] + "  " +str(item[-2]), (0, 255, 0))
        # sym = read_sym_name(map_file, item[-1])
        sym = syms[int(item[-1])]
        draw.text([xmin, ymin], sym + "  " +str(item[-2]), (0, 255, 0))
        # print item, sym
        result_box = [xmin, ymin, xmax, ymax]
        result_box_all.append(result_box)
        # print [xmin, ymin, xmax, ymax]
        # print [xmin, ymin], item[-1]

        # print("Top_5 of result: ")
        box_sym_top5 = []
        box_conf_top5 = []
        for j in range(5):
        	sym_idx = det_top_5_label[i][j] - 1
        	conf = det_top_5_conf[i][j]
        	sym = syms[sym_idx]
        	# print "top_{}: [sym:{} conf:{}]".format(str(j+1), sym, str(conf))
        	det_str = "top_{}: [sym:{} conf:{}]".format(str(j+1), sym, str(conf))
        	# print(det_str)
        	box_sym_top5.append(sym)
        	box_conf_top5.append(conf)

        result_sym_top5.append(box_sym_top5)
        result_conf_top5.append(box_conf_top5)
        
    # time2 = time.time() - start
    # print(time2, time1)
    img.save('result/detect_result.jpg')
    # img_sym = cv2.imread('result/detect_result.jpg')
    # cv2.imshow("sym detection", img_sym)
    # cv2.waitKey()

    return result_box_all, result_sym_top5, result_conf_top5


def read_sym_name(map_file, sym_index=None):
	with open(map_file, 'r') as mf:
		data = mf.readlines()
	syms = []
	for line in data:
		sym = line.split('\n')[0].strip()
		syms.append(sym)
	if sym_index:
		sym = syms[int(sym_index)]
		return sym
	else:
		return syms


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='model/labelmap_voc_sym.prototxt')
    parser.add_argument('--model_def',
                        default='model/deploy.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_weights',
                        default='model/VGG_three_grade_SSD_300x300_iter_85000.caffemodel')
    parser.add_argument('--image_file', default='images/formula_ssd.jpg')
    parser.add_argument('--map_file', default='map_file/grade_three_map.txt')
    return parser.parse_args()


if __name__ == '__main__':
	# map_file = "/home/nd/project/object_detect/caffe/examples/ssd/grade_three_map.txt"
    det_result = main(parse_args())
