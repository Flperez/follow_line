#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
import threading
import time
import cv2
from datetime import datetime
import time
import operator
import math as mt

time_cycle = 80

class MyAlgorithm(threading.Thread):




    def __init__(self, cameraL, cameraR, motors, pose_client):
        self.cameraL = cameraL
        self.cameraR = cameraR
        self.pose_client = pose_client
        self.motors = motors
        self.imageRight=None
        self.imageLeft=None

        self.ref3point = np.array(([317-1, 250], [287-52, 364], [262-109, 479]), dtype=np.int)
        self.previous_error = np.array([0,0,0],dtype = np.int)
        self.state="straight"

        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)

    def setRightImageFiltered(self, image):
        self.lock.acquire()
        self.imageRight=image
        self.lock.release()


    def setLeftImageFiltered(self, image):
        self.lock.acquire()
        self.imageLeft=image
        self.lock.release()

    def getRightImageFiltered(self):
        self.lock.acquire()
        tempImage=self.imageRight
        self.lock.release()
        return tempImage

    def getLeftImageFiltered(self):
        self.lock.acquire()
        tempImage=self.imageLeft
        self.lock.release()
        return tempImage

    def run (self):

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.execute()
            finish_Time = datetime.now()
            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()


    def createmask(self,image):

        ###### Mask ######
        blur = cv2.GaussianBlur(image, (5, 5), 1)
        hsv = cv2.cvtColor(src=blur, code=cv2.COLOR_RGB2HSV)
        min_values = [0,100,50]
        max_values = [0,255,255]


        # hsv = image
        lower_red_hue_range = cv2.inRange(src=hsv, lowerb=np.array(min_values), upperb=np.array(max_values))
        mask = cv2.erode(lower_red_hue_range,(7,7),iterations=5)
        #clower_red_hue_range = cv2.inRange(src=hsv, lowerb=np.array([120, 200, 10]), upperb=np.array([170, 255, 179]))
        # upper_red_hue_range = cv2.inRange(src=hsv, lowerb=np.array([160, 100, 100]), upperb=np.array([179, 255, 255]))
        # return cv2.addWeighted(src1=lower_red_hue_range, alpha=1, src2=upper_red_hue_range, beta=1, gamma=0)
        return lower_red_hue_range
        # return upper_red_hue_range
    def getMycontour(self, contours):
        limits = [np.max(contours[i][:][:, 0, 1]) for i in range(len(contours))]
        shapes = [contours[i].shape[0] for i in range(len(contours))]
        limits_shapes = zip(limits, shapes)
        limits_shapes_sort = sorted(limits_shapes, key=operator.itemgetter(1), reverse=True)
        # Entre los dos con mas puntos, elige el que tenga un punto en y mayor
        maximo = max(zip(*limits_shapes_sort[:2])[0])

        return limits.index(maximo)

    def getMaxcontour(self,contours):
        lst=[]
        if contours:
            for idx,cnt in enumerate(contours):
                cnt = np.reshape(cnt,(-1,2))
                lst.append((np.max(cnt[:,1]),cnt.shape[0],idx))
            lst_zip = np.asarray(sorted(lst, key=operator.itemgetter(1), reverse=True)[:3])
            return contours[lst_zip[np.argmax(lst_zip[:,0]),2]]
        else:
            return  np.array([])


    def getNpoint(self,mycontour,N):
        height = 480
        ejeY = np.linspace(np.min(np.min(mycontour[:,0,1])),np.max(mycontour[:,0,1])-1,N).astype(int)
        # print mycontour.shape
        # for line in ejeY:
        #     # print ejeY
        #     print np.where(mycontour[:,0,1]==line)
        try:
            listmax = [np.max(np.where(mycontour[:,0,1]==line)) for line in ejeY]
            listmin = [np.min(np.where(mycontour[:,0,1]==line)) for line in ejeY]
            ejeX = 0.5*(mycontour[listmax,0,0] + mycontour[listmin,0,0])
            return np.concatenate((np.vstack(ejeX), np.vstack(ejeY)), 1).astype(int)

        except ValueError:
            return np.array([])



    def drawNpoint(self,image,points,color):
        for i in range(len(points)):
            cv2.circle(image,tuple(points[i]),5,color,4)
        return image

    def drawError(self,image,reference,points):
        for i in range(len(points)):
            cv2.line(image,tuple(reference[i]),tuple(points[i]),(0,255,0))
        return image

    def getError(self,reference,points):
        return (reference-points)[:,0]

    def getState(self,previous_error,actual_error):
        inc_error = actual_error-previous_error
        if actual_error[1]>50:
            self.state = "left"
        elif actual_error[1]<-50:
            self.state = "right"
        else:
            self.state = "straight"
        return self.state,inc_error





    def stateOfMachine(self,state,actual_error,inc_error):
        if state == "straight":
            vel = 0.2
            w = 0
        elif state == "right":
            vel = 0.2
            K = 0.01
            w = K * inc_error[1]
        else:
            K = 0.01
            w = - K * inc_error[1]
            vel = 0.2


        self.motors.sendV(vel)
        self.motors.sendW(w)



    def execute(self):
        #GETTING THE IMAGES
        #imageLeft = self.cameraL.getImage()
        imageRight = self.cameraR.getImage()
        self.pose = self.pose_client.getPose3d()

        # Add your code here
        start = time.clock()

        ###### Limitamos la imagen de entrada
        imageRight = imageRight.data
        imageRightdata = np.zeros([480, 640, 3], dtype=np.uint8)
        procces = np.zeros([480, 640, 3], dtype=np.uint8)
        imageRightdata[480 / 2:, :] = imageRight[480 / 2:, :]

        mask = MyAlgorithm.createmask(self,imageRightdata)
        maskcolor = cv2.cvtColor(src=mask, code=cv2.COLOR_GRAY2RGB)
        self.setLeftImageFiltered(maskcolor)

        # print time.clock() - start
        # cv2.imshow("jajaja", maskcolor)
        # cv2.waitKey(1)

        #
        #
        _,contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mycnt = MyAlgorithm.getMaxcontour(self,contours)
        if mycnt != np.array([]):
            cv2.drawContours(procces, mycnt, -1, [255, 0, 0])
            Npoint = MyAlgorithm.getNpoint(self,mycontour=mycnt,N=3)
            if Npoint != np.array([]):
                procces = MyAlgorithm.drawNpoint(self,procces, Npoint, (255,0,0))
        #     procces = MyAlgorithm.drawNpoint(self,procces, self.ref3point, (0,0,255))
        #     procces = MyAlgorithm.drawError(self,procces,reference=self.ref3point,points=Npoint)
        #     self.actual_error = MyAlgorithm.getError(self,self.ref3point,Npoint)
        #     if self.previous_error != np.array([]):
        #         state,inc_error = MyAlgorithm.getState(self,self.previous_error,self.actual_error)
        #         MyAlgorithm.stateOfMachine(self, state, self.actual_error, inc_error)
        #         print inc_error,state
        #
        #     self.previous_error = self.actual_error



        ###### Convertimos la mascara a rgb para visualizarla

        #SHOW THE FILTERED IMAGE ON THE GUI
        self.setRightImageFiltered(procces)
