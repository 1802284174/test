from typing import List

from PyQt5 import QtCore, QtGui, QtWidgets
from rknnlite.api import RKNNLite
from coco_utils import COCO_test_helper
import sys
import cv2
import numpy as np
import time

CLASSES = ("baby", "climbing", "fall", "lay", "faint")
class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer_camera = QtCore.QTimer() # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture() # 视频流
        self.CAM_NUM = 21  # 为0时表示视频流来自笔记本内置摄像头
        self.set_ui() # 初始化程序界面
        self.slot_init() # 初始化槽函数



    def set_ui(self):
        self.layout_main = QtWidgets.QHBoxLayout() # 总布局
        self.layout_fun_button = QtWidgets.QVBoxLayout() # 按键布局
        self.layout_data_show = QtWidgets.QVBoxLayout() # 数据(视频)显示布局
        self.button_open_camera = QtWidgets.QPushButton('打开相机') # 建立用于打开摄像头的按键
        self.button_close = QtWidgets.QPushButton('退出') # 建立用于退出程序的按键
        self.button_open_camera.setMinimumHeight(50) # 设置按键大小
        self.button_close.setMinimumHeight(50)
        self.button_test = QtWidgets.QPushButton('测试')
        self.label_show_camera = QtWidgets.QLabel() # 定义显示视频的Label
        self.label_show_camera.setFixedSize(800, 600) # 给显示视频的Label设置大小为800x600
        self.layout_fun_button.addWidget(self.button_open_camera) # 把打开摄像头的按键放到按键布局中
        self.layout_fun_button.addWidget(self.button_close) # 把退出程序的按键放到按键布局中
        self.layout_fun_button.addWidget(self.button_test)
        self.layout_main.addLayout(self.layout_fun_button) # 把按键布局加入到总布局中
        self.layout_main.addWidget(self.label_show_camera) # 把用于显示视频的Label加入到总布局中
        self.setLayout(self.layout_main) # 设置总布局

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked) # 若该按键被点击，则调用button_open_camera_clicked()
        self.timer_camera.timeout.connect(self.show_camera) # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close) # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
        self.button_test.clicked.connect(self.test)

    def button_open_camera_clicked(self):
        if not self.timer_camera.isActive(): # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM) # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频

            if not flag: # flag表示open()成不成功
                QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30) # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
        else:
            self.timer_camera.stop() # 关闭定时器
            self.cap.release() # 释放视频流
            self.label_show_camera.clear() # 清空视频显示区域
            self.button_open_camera.setText('打开相机')
    def test(self):
        if not self.timer_camera.isActive():  # 若定时器未启动
            flag = self.cap.open('/home/elf/work/test.mp4')  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
            self.button_test.setText('测试')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.buttontest.setText('测试')
    def show_camera(self):
        rknn = RKNNLite(verbose=0)
        print('加载 RKNN 模型...')
        ret = rknn.load_rknn('/home/elf/work/yolov5.rknn')
        if ret != 0:
            print('加载模型失败！')
            exit(ret)
        print('加载模型成功！')
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret == 0:
            print('init runtime success!')
        print('init runtime failed!')
        global OBJ_THRESH, NMS_THRESH, IMG_SIZE
        OBJ_THRESH = 0.5
        NMS_THRESH = 0.65
        ANCHORS = [
            [[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]],
            [[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]],
            [[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]]
        ]
        IMG_SIZE = tuple([640, 640])
        fps = 0.0
        while True:
            t1 = time.time()
            falg, self.image = self.cap.read()  # 从视频流中读取
            co_helper = COCO_test_helper(enable_letter_box=True)
            show = self.image
            img = co_helper.letter_box(im=show, new_shape=(640, 640), pad_color=(0, 0, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 视频色彩转换回RGB，这样才是现实的颜色
            img = np.expand_dims(img,axis=0)
            outputs = rknn.inference(inputs=[img])
            if outputs is None:
                print("推理失败，输出为 None！")
                rknn.release()
            boxes, classes, scores = post_process(outputs,ANCHORS)# 传递 anchors 参数
            img_1 = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)
            if boxes is not None:
                draw(img_1, boxes, scores, classes, CLASSES)
                img_1 =  cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(img_1.data, img_1.shape[1], img_1.shape[0],QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                rknn.release()
            else:
                show = co_helper.letter_box(im=show, new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                rknn.release()
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage)) # 往显示视频的Label里显示QImage

def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh):
    """根据对象阈值过滤框"""
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= obj_thresh)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores, nms_thresh):
    """非极大值抑制"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)

def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] //grid_w]).reshape(1, 2, 1, 1)
    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)
    box_xy = position[:, :2, :, :] * 2 - 0.5
    box_wh = pow(position[:, 2:4, :, :] * 2, 2) * anchors
    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2  # bottom right y
    return xyxy

def post_process(input_data,anchors):
    boxes, scores, classes_conf = [], [], []
    input_data = [_in.reshape([len(anchors[0]), -1] + list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:, :4, :, :], anchors[i]))
        scores.append(input_data[i][:, 4:5, :, :])
        classes_conf.append(input_data[i][:, 5:, :, :])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # 过滤和 NMS
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf, OBJ_THRESH)
    if len(boxes) == 0:
        return None, None, None

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s, NMS_THRESH)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

def draw_box_corner(draw_img, top, left, right, bottom, length, corner_color):
    # Top Left
    cv2.line(draw_img, (top, left), (top + length, left), corner_color, thickness=3)
    cv2.line(draw_img, (top, left), (top, left + length), corner_color, thickness=3)
    # Top Right
    cv2.line(draw_img, (right, left), (right - length, left), corner_color, thickness=3)
    cv2.line(draw_img, (right, left), (right, left + length), corner_color, thickness=3)
    # Bottom Left
    cv2.line(draw_img, (top, bottom), (top + length, bottom), corner_color, thickness=3)
    cv2.line(draw_img, (top, bottom), (top, bottom - length), corner_color, thickness=3)
    # Bottom Right
    cv2.line(draw_img, (right, bottom), (right - length, bottom), corner_color, thickness=3)
    cv2.line(draw_img, (right, bottom), (right, bottom - length), corner_color, thickness=3)

def draw_label_type(draw_img, top, left, label, label_color):
    labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 6)[0]
    if left - labelSize[1] - 3 < 0:
        box_coords = (top, left + 5, top + labelSize[0], left + labelSize[1] + 3)
        text_pos = (top, left + labelSize[0] + 3)
    else:
        box_coords = (top, left - labelSize[1] - 3, top + labelSize[0], left - 3)
        text_pos = (top, left - 3)
    cv2.rectangle(draw_img, box_coords[0:2], box_coords[2:4], color=label_color, thickness=-1)
    cv2.putText(draw_img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

def draw(image, boxes, scores, classes, classes_list):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 255), 2)
        draw_box_corner(image, top, left, right, bottom, 15, (0, 255, 255))
        draw_label_type(image, top, left, f"{classes_list[cl]} {score:.2f}", (255, 0, 255))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) # 固定的，表示程序应用
    ui = Ui_MainWindow() # 实例化Ui_MainWindow
    ui.show() # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_()) # 不加这句，程序界面会一闪而过