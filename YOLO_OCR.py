import copy
import math
import torch

from PIL import Image, ImageDraw, ImageFont
from darknet import Darknet
from tools.gen_image import draw
from utils import nms, get_region_boxes, load_class_names, get_region_boxes_topk, plot_boxes
import numpy as np


class YOLO2_OCR():
    def __init__(self, cfg_file, weight_file, label_file, use_cuda=True, image_size=800, formule_pad=200):
        self.label_file = label_file
        self.m = Darknet(cfg_file)
        self.m.print_network()
        self.m.load_weights(weight_file)
        self.image_size = image_size
        self.formule_pad = formule_pad
        print('Loading weights from %s... Done!' % (weight_file))

        if use_cuda:
            print("Using GPU."),
            self.m.cuda()
        else:
            print("Using CPU."),
            
    def get_canv_bbox(self, pts_list, boxes):
        xs = []
        ys = []
        #for pt in pts_list:
        for i in range(0, len(pts_list)/2, 1):
            pt = [pts_list[2*i], pts_list[2*i+1]]
            if pt[0] != -10000 and pt[1] != -10000:
                xs.append(int(pt[0]))
                ys.append(int(pt[1]))
    
        np_xs = np.array(xs).argsort()
        np_ys = np.array(ys).argsort()
    
        min_x = xs[np_xs[0]]
        min_y = ys[np_ys[0]]
        max_x = xs[np_xs[-1]]
        max_y = ys[np_ys[-1]]
    
        trace_w = max_x - min_x + 1
        trace_h = max_y - min_y + 1
        if trace_w < 0 or trace_h < 0:
            assert ("pts error.")
    
        canv_boxes = []
        for box in boxes:
            im_x1 = box[0]
            im_y1 = box[1]
            im_x2 = box[2]
            im_y2 = box[3]
    
            canv_x1 = min_x + im_x1 * float((max(trace_w, trace_h) + self.formule_pad)) / self.image_size - float((
                        max(trace_w, trace_h) + self.formule_pad - trace_w)) / 2 -3
    
            canv_y1 = min_y + im_y1 * float((max(trace_w, trace_h) + self.formule_pad)) / self.image_size - float((
                        max(trace_w, trace_h) + self.formule_pad - trace_h)) / 2 -3
    
            canv_x2 = min_x + im_x2 * float((max(trace_w, trace_h) + self.formule_pad)) / self.image_size - float((
                        max(trace_w, trace_h) + self.formule_pad - trace_w)) / 2 +5
    
            canv_y2 = min_y + im_y2 * float((max(trace_w, trace_h) + self.formule_pad)) / self.image_size - float((
                        max(trace_w, trace_h) + self.formule_pad - trace_h)) / 2 +5
            canv_boxes.append([canv_x1, canv_y1, canv_x2, canv_y2])
    
        return canv_boxes


    def reference(self, trace_seq, topk=5, use_cuda=True):
        '''

        Args:
            trace_seq: [300, 200, 300, 400, -10000, -10000, 800, 200, 800, 400, -10000, -10000, ...]
            use_cuda:

        Returns:

        '''

        traces = self.process_trace(trace_seq)
        draw(traces, image_size=self.image_size, formule_pad=self.formule_pad)

        imgfile = "pad_formule.jpg"
        img = Image.open(imgfile).convert('RGB')
        sized = img.resize((self.m.width, self.m.height))

        # for i in range(2):
        # start = time.time()
        boxes, topk_cls_list, topk_cls_conf = self.do_detect(self.m, sized, 0.5, 0.4, topk, use_cuda)  # forward
        # finish = time.time()
        # if i == 1:
        # print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        class_names = load_class_names(self.label_file)
        print(class_names)
        savename, ret_boxes, ret_syms, ret_cls_conf = plot_boxes(img, boxes, 'predictions.jpg', class_names)
        print(ret_syms)

        print "========= top-k ======="
        topk_syms = []
        topk_cls = []
        topk_syms_tmp = []
        topk_cls_tmp = []
        for i in range(len(topk_cls_list)):
            for j in range(topk):
                print "class_name: %s, cls_conf: %.6f" % (class_names[topk_cls_list[i][j]], topk_cls_conf[i][j])
                topk_syms_tmp.append(class_names[topk_cls_list[i][j]])
                topk_cls_tmp.append(topk_cls_conf[i][j])
            print
            topk_syms.append(topk_syms_tmp)
            topk_cls.append(topk_cls_tmp)
            topk_syms_tmp = []
            topk_cls_tmp = []

        print "======================="
        print(traces)
        ret_boxes = self.get_canv_bbox(trace_seq, ret_boxes)

        return savename, ret_boxes, ret_syms, ret_cls_conf, topk_syms, topk_cls
        # return dets


    def detect_labels(self, img, boxes, class_names):
        dets = []
        width = img.width
        height = img.height

        for i in range(len(boxes)):
            box = boxes[i]
            x1 = (box[0] - box[2]/2.0) * width
            y1 = (box[1] - box[3]/2.0) * height
            x2 = (box[0] + box[2]/2.0) * width
            y2 = (box[1] + box[3]/2.0) * height

            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print('%s: %f' % (class_names[cls_id], cls_conf))
                dets.append([x1, y1, x2, y2, cls_conf, class_names[cls_id]])

        return dets


    def do_detect(self, model, img, conf_thresh, nms_thresh, topk=5, use_cuda=True):
        model.eval()
        # t0 = time.time()

        if isinstance(img, Image.Image):
            width = img.width
            height = img.height
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
            img = img.view(1, 3, height, width)
            img = img.float().div(255.0)
        elif type(img) == np.ndarray: # cv2 image
            img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
        else:
            print("unknow image type")
            exit(-1)

        # t1 = time.time()

        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)
        # t2 = time.time()

        output = model(img)
        output = output.data
        #for j in range(100):
        #    sys.stdout.write('%f ' % (output.storage()[j]))
        #print('')
        # output = output.cpu()
        # t3 = time.time()
        boxes, anchor_idxs_topk, cls_topk_confs, cls_topk_ids = get_region_boxes_topk(output, conf_thresh,
                                                                                      model.num_classes, model.anchors,
                                                                                      model.num_anchors, topk=topk,
                                                                                      use_cuda=use_cuda)
        boxes = boxes[0]

        # NMS
        boxes, nms_left_boxids = nms(boxes, nms_thresh)

        #
        topk_anchor_ids = []
        topk_cls_list = []
        topk_cls_conf = []
        for k in nms_left_boxids:
            topk_anchor_ids.append(anchor_idxs_topk[k])
        print "anchor idx left:", topk_anchor_ids

        for idx in topk_anchor_ids:
            topk_cls_list.append(cls_topk_ids[idx])
            topk_cls_conf.append(cls_topk_confs[idx])
        return boxes, topk_cls_list, topk_cls_conf

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
    cfg_file = "../../cfg/yolo-voc-ocr-test.cfg"
    weight_file = "../../models/model_yolo2/000050.weights"
    label_file = "../../data/charmap_split_sqrt.txt"

    model = YOLO2_OCR(cfg_file, weight_file, label_file)

    pt_seq = [300, 200, 300, 400, -10000, -10000, 800, 200, 800, 400, -10000, -10000]

    #trace_file = "/home/hyer/Downloads/99U_Downloads/2020-math-math-2018-01-17-17-59-25-064794-192fa79a-fb6d-11e7-b9c5-9c5c8e8f37ca.txt"
    #with open(trace_file, "r") as f:
    #    data = f.readlines()

    #pt_seq = []
    #for li in data:
    #    line = li.strip()
    #    # if line != "-10000 -10000\n":
    #    pt_seq.append(int(float(line.split()[0])))
    #    pt_seq.append(int(float(line.split()[1])))

    savename, ret_boxes, ret_syms, ret_cls_conf, topk_syms, topk_cls = model.reference(pt_seq)
    print "xxx"



class Namelist(list):
	"""docstring for Namelist"""
	def __init__(self, a_name):
		super(Namelist, self).__init__()
		self.arg = arg
		
