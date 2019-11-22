#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')


def center_point_inside_polygon(bounding_box, polygon_coord):
    center = (int(bounding_box[0] + (bounding_box[2]) / 2), int(bounding_box[1] + (bounding_box[3]) / 2))
    polygon_coord = np.array(polygon_coord, np.int32)
    polygon_coord = polygon_coord.reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(polygon_coord, center, False)
    # In the pointPolygonTest function, third argument is measureDist. If it is True, it finds the signed distance.
    # If False, it finds whether the point is inside or outside or on the contour
    # (it returns +1, -1, 0 respectively)
    if result == 1:
        return result
    else:
        return 0

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('/home/harshita/Downloads/IP\ Camera8_Blr_20191121163519_20191121163647_796109.mp4')

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('op_test.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        print(type(frame) , " is the type of image")
        #Adding region info for door and counter 
        pts = np.array([[841.5,141.5],[1009.5,166.5],[1018.5,318],[742.5,288.0]], np.int32)
        pts = pts.reshape((-1,1,2))
        pts2 = np.array([[742.5,288.0],[1053.0,318.0],[1065.0,702.0],[544.5,679.5]], np.int32)
        pts2 = pts2.reshape((-1,1,2))
        cv2.polylines(frame,[pts],True,(0,255,255))
        cv2.polylines(frame,[pts2],True,(255,0,255))
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            #Person detection
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

            if center_point_inside_polygon(bbox,pts):
                #Check if the person is at counter
                #Trackid number
                cv2.putText(frame, str(track.track_id) + str("_at_counter"),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            elif center_point_inside_polygon(bbox,pts2):
                #Trackid number
                #Check if the person is at door
                cv2.putText(frame, str(track.track_id) + str("_at_door"),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            else:
                #Trackid number
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2) 
                

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
