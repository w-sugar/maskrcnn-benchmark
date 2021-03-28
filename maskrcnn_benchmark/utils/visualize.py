#!/usr/bin/env python

import os.path
import torch
import sys
# import socket
import time
# import traceback
# from urlparse import urlparse
# import cStringIO
# import PIL.Image
# from PIL import Image
import numpy as np
from collections import OrderedDict
# import cv2
import math

# from visdom import Visdom
# vis = Visdom(server='http://127.0.0.1', port=8097)

all_wins = {}


def clear_wins(vis):
    pass


def print_dict(vis, loss_dict, iters, need_plot=False):
    # data_str = ''
    for k, v in loss_dict.items():
        # if data_str != '':
        #     data_str += ', '
        # data_str += '{}: {:.10f}'.format(k, v.item())

        if need_plot and vis is not None:
            plot_single_line(vis, k, k, iters, v.item())



def plot_single_line(vis, title, name, x, v):
    win = all_wins.get(title, None)
    if win is None:
        win = vis.line(env='main', X=np.array([x]), Y=np.array([v]), opts={'legend':[name], 'title':title})
        all_wins[title] = win
    else:
        # vis.updateTrace(env='main', win=win,  X=np.array([i]), Y=np.array([v]), name=name)
        vis.line(env='main', win=win, X=np.array([x]), Y=np.array([v]), update='append', opts={'legend':[name], 'title':title})



def plot_coco_aps(vis, title, value_dict, iters):
    win = all_wins.get(title, None)
    apname_list = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
    x_list = [iters, iters, iters, iters, iters, iters]
    if vis is not None:
        values = []
        for k in apname_list:
            values.append(value_dict[k])

        if win is None:
            win = vis.line(env='main', X=np.array([iters]), Y=np.array([values]), 
                           opts={'legend':apname_list, 'title':title, 'markersize':30})
            all_wins[title] = win
        else:
            vis.line(env='main', win=win, X=np.array([x_list]), Y=np.array([values]), update='append', 
                     opts={'legend':apname_list, 'title':title, 'markersize':30})


def plot_coco_mr_crowdhuman(vis, title, value_dict, iters):
    win = all_wins.get(title, None)
    apname_list = ['Reasonable', 'Small', 'Heavy', 'All']
    x_list = [iters, iters, iters, iters]
    if vis is not None:
        values = []
        for k in apname_list:
            values.append(value_dict[k])

        if win is None:
            win = vis.line(env='main', X=np.array([iters]), Y=np.array([values]), 
                           opts={'legend':apname_list, 'title':title, 'markersize':30})
            all_wins[title] = win
        else:
            vis.line(env='main', win=win, X=np.array([x_list]), Y=np.array([values]), update='append', 
                     opts={'legend':apname_list, 'title':title, 'markersize':30})


def plot_coco_mr_citypersons(vis, title, value_dict, iters):
    win = all_wins.get(title, None)
    apname_list = ['Reasonable', 'Bare', 'Partial', 'Heavy']
    x_list = [iters, iters, iters, iters]
    if vis is not None:
        values = []
        for k in apname_list:
            values.append(value_dict[k])

        if win is None:
            win = vis.line(env='main', X=np.array([iters]), Y=np.array([values]), 
                           opts={'legend':apname_list, 'title':title, 'markersize':30})
            all_wins[title] = win
        else:
            vis.line(env='main', win=win, X=np.array([x_list]), Y=np.array([values]), update='append', 
                     opts={'legend':apname_list, 'title':title, 'markersize':30})



def plot_voc_map(vis, title, map_val, iters):
    win = all_wins.get(title, None)
    if win is None:
        win = vis.line(env='main', X=np.array([iters]), Y=np.array([map_val]), opts={'legend':['mAP'], 'title':title, 'linecolor': np.array([[255,0,0]])})
        all_wins[title] = win
    else:
        vis.line(env='main', win=win, X=np.array([iters]), Y=np.array([map_val]), update='append', opts={'legend':['mAP'], 'title':title, 'linecolor': np.array([[255,0,0]])})



def plot_voc_aps(vis, title, aps, iters):
    win = all_wins.get(title, None)
    cat_list = []
    ap_list = []
    x_list = []
    for cat_name, ap in aps.items():
        cat_list.append(cat_name)
        ap_list.append(ap)
        x_list.append(iters)
    if vis is not None:
        if win is None:
            win = vis.line(env='main', X=np.array([iters]), Y=np.array([ap_list]), 
                           opts={'legend':cat_list, 'title':title, 'markersize':30})
            all_wins[title] = win
        else:
            vis.line(env='main', win=win, X=np.array([x_list]), Y=np.array([ap_list]), update='append', 
                     opts={'legend':cat_list, 'title':title, 'markersize':30})




# class AverageWithinWindow():
#     def __init__(self, win_size):
#         self.win_size = win_size
#         self.cache = []
#         self.average = 0
#         self.count = 0

#     def update(self, v):
#         if self.count < self.win_size:
#             self.cache.append(v)
#             self.count += 1
#             self.average = (self.average * (self.count - 1) + v) / self.count
#         else:
#             idx = self.count % self.win_size
#             self.average += (v - self.cache[idx]) / self.win_size
#             self.cache[idx] = v
#             self.count += 1


# class DictAccumulator():
#     def __init__(self, win_size=None):
#         self.accumulator = OrderedDict()
#         self.total_num = 0 
#         self.win_size = win_size

#     def update(self, d):
#         self.total_num += 1
#         for k, v in d.items():
#             if not self.win_size:
#                 self.accumulator[k] = v + self.accumulator.get(k,0)
#             else:
#                 self.accumulator.setdefault(k, AverageWithinWindow(self.win_size)).update(v)

#     def get_average(self):
#         average = OrderedDict()
#         for k, v in self.accumulator.items():
#             if not self.win_size:
#                 average[k] = v*1.0/self.total_num 
#             else:
#                 average[k] = v.average 
#         return average


