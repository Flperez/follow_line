#!/usr/bin/python
# -*- coding: utf-8 -*-
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
        self.imageRight = None
        self.imageLeft = None

        self.ref3point = np.array(([316, 244], [235, 361], [153, 478]), dtype=np.int)
        self.m = float(self.ref3point[2, 1] - self.ref3point[0, 1]) / float(
            (self.ref3point[2, 0] - self.ref3point[0, 0]))
        self.n = self.ref3point[0, 1] - self.m * self.ref3point[0, 0]

        self.calc_previous_error = None
        self.state = "straight"

        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)

    def setRightImageFiltered(self, image):
        self.lock.acquire()
        self.imageRight = image
        self.lock.release()

    def setLeftImageFiltered(self, image):
        self.lock.acquire()
        self.imageLeft = image
        self.lock.release()

    def getRightImageFiltered(self):
        self.lock.acquire()
        tempImage = self.imageRight
        self.lock.release()
        return tempImage

    def getLeftImageFiltered(self):
        self.lock.acquire()
        tempImage = self.imageLeft
        self.lock.release()
        return tempImage

    def run(self):

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.execute()
            finish_Time = datetime.now()
            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            # print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop(self):
        self.stop_event.set()

    def play(self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill(self):
        self.kill_event.set()

    def createmask(self, image):

        ###### Mask ######
        blur = cv2.GaussianBlur(image, (5, 5), 1)
        hsv = cv2.cvtColor(src=blur, code=cv2.COLOR_BGR2HSV)
        min_values = [120, 200, 10]
        max_values = [170, 255, 180]

        # hsv = image
        lower_red_hue_range = cv2.inRange(src=hsv, lowerb=np.array(min_values), upperb=np.array(max_values))
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

    def getMaxcontour(self, contours):
        lst = []
        if contours:
            for idx, cnt in enumerate(contours):
                cnt = np.reshape(cnt, (-1, 2))
                lst.append((np.max(cnt[:, 1]), cnt.shape[0], idx))
            lst_zip = np.asarray(sorted(lst, key=operator.itemgetter(1), reverse=True)[:3])
            return contours[lst_zip[np.argmax(lst_zip[:, 0]), 2]]
        else:
            return np.array([])

    def getNpoint(self, mycontour, N):
        height = 480
        ejeY = np.array([np.min(mycontour[:, 0, 1]), np.min(mycontour[:, 0, 1]) + 50, np.min(mycontour[:, 0, 1]) + 70])

        try:
            listmax = [np.max(np.where(mycontour[:, 0, 1] == line)) for line in ejeY]
            listmin = [np.min(np.where(mycontour[:, 0, 1] == line)) for line in ejeY]
            ejeX = 0.5 * (mycontour[listmax, 0, 0] + mycontour[listmin, 0, 0])
            return np.concatenate((np.vstack(ejeX), np.vstack(ejeY)), 1).astype(int)

        except ValueError:
            return np.array([])

    def drawNpoint(self, image, points, color):
        for i in range(len(points)):
            cv2.circle(image, tuple(points[i]), 5, color, 4)
        return image

    def drawError(self, image, reference, points):
        for i in range(len(points)):
            cv2.line(image, tuple(reference[i]), tuple(points[i]), (0, 255, 0))
        return image

    def getError(self, points, alfa=0.6, beta=0.0, gamma=0.4):
        reference = np.concatenate((np.asarray(map(lambda x: (x[1] - self.n) / self.m, points.tolist()),
                                               dtype=np.int).reshape(-1, 1), points[:, 1].reshape(-1, 1)), axis=1)
        error = (reference - points)[:, 0]
        return error, MyAlgorithm.calcError(self, error, alfa, beta, gamma), reference

    def getState(self, actual_error):

        if actual_error[0] < -50 and actual_error[2] > 50:
            self.state = "right"
        elif actual_error[0] > 50 and actual_error[2] < -50:
            self.state = "left"
        elif actual_error[0] < -50:
            self.state = "right"
        elif actual_error[0] > 50:
            self.state = "left"

        elif (self.state != "right" and self.state != "left") and -30<actual_error[0] < 0 and abs(actual_error[2])-abs(actual_error[0]) > 80:
            self.state = "out curveR"

        elif ( self.state != "right" and self.state != "left") and -0<actual_error[0] < 30 and abs(actual_error[2])-abs(actual_error[0]) > 80:
            self.state = "out curveL"

        elif (actual_error[0] < -10) and (self.state == "straight" or self.state == "close curveR" or self.state == "out curveL" or self.state == "out curveR") and abs(actual_error[2])-abs(actual_error[0]) < 100:
            self.state = "close curveR"

        elif (actual_error[0] > 10) and (self.state == "straight" or self.state == "close curveL" or self.state == "out curveL" or self.state == "out curveR") and abs(actual_error[2])-abs(actual_error[0]) < 100:
            self.state = "close curveL"
        else:
            self.state = "straight"

        return self.state

    def calcError(self, error, alfa, beta, gamma):
        return alfa * abs(error[0]) + beta * abs(error[1]) + gamma * abs(error[2])

    def getStatusImage(self, calc_error_actual, actual_error, calc_error_inc, state, vel, w):
        out = np.zeros([480, 640, 3], dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, "State: %s" % (state), (10, 150), font, 1, (255, 255, 255))
        cv2.putText(out, "v: %f   w: %f" % (vel, w), (10, 200), font, 1, (255, 0, 255))
        cv2.putText(out, "CalcErr: %2.f" % (calc_error_actual), (10, 250), font, 1, (255, 255, 0))
        cv2.putText(out, "Err: %2.f %2.f %2.f" % (actual_error[0], actual_error[1], actual_error[2]), (10, 300), font,
                    1, (255, 255, 0))
        cv2.putText(out, "inc: %2.f" % (calc_error_inc), (10, 350), font, 1, (255, 255, 0))
        return out

    def stateOfMachine(self, state, actual_error, calc_error_actual, calc_previous_error):
        calc_error_inc = abs(calc_error_actual - calc_previous_error)
        # print calc_error_actual - calc_previous_error

        velMax = 10.5

        # # Disacelerate
        if state == "close curveR":

            vel = (1 - float(abs(actual_error[0])) / 100) * 2 + 7.5
            Kwp = -0.0005
            Kwd = -0.00025
            w = Kwp * calc_error_actual + Kwd * calc_error_inc
        elif state == "close curveL":
            vel = (1 - float(abs(actual_error[0])) / 100) * 2 + 7.5
            Kwp = 0.0005
            Kwd = 0.00025
            w = Kwp * calc_error_actual + Kwd * calc_error_inc

        elif state == "straight":
            vel = velMax
            w = 0

        elif state == "right" or state == "strong right":
            Kvp = 0.005
            Kvd = 0.003
            Kwp = -0.0055
            Kwd = -0.004
            vel = 6.65 - Kvp * calc_error_actual - Kvd * calc_error_inc
            w = Kwp * calc_error_actual + Kwd * calc_error_inc
        elif state == "left" or state == "strong left":

            Kvp = 0.005
            Kvd = 0.003
            Kwp = 0.0055
            Kwd = 0.0045
            vel = 6.65 - Kvp * calc_error_actual - Kvd * calc_error_inc

            w = Kwp * calc_error_actual + Kwd * calc_error_inc
        elif state == "out curveR":
            vel = 10
            Kwp = -0.0004
            Kwd = -0.0003
            w = Kwp * calc_error_actual + Kwd * calc_error_inc

        elif state == "out curveL":
            vel = 10
            Kwp = 0.0004
            Kwd = 0.0003
            w = Kwp * calc_error_actual + Kwd * calc_error_inc

        self.motors.sendV(vel)
        self.motors.sendW(w)
        return calc_error_actual, vel, w, calc_error_inc

    def execute(self):
        # GETTING THE IMAGES
        # imageLeft = self.cameraL.getImage()
        imageRight = self.cameraR.getImage()
        self.pose = self.pose_client.getPose3d()

        # Add your code here
        start = time.clock()

        ###### Limitamos la imagen de entrada
        imageRight = imageRight.data
        imageRightdata = np.zeros([480, 640, 3], dtype=np.uint8)
        procces = np.zeros([480, 640, 3], dtype=np.uint8)
        imageRightdata[480 / 2:, :] = imageRight[480 / 2:, :]

        mask = MyAlgorithm.createmask(self, imageRightdata)
        maskcolor = cv2.cvtColor(src=mask, code=cv2.COLOR_GRAY2RGB)

        _, contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mycnt = MyAlgorithm.getMaxcontour(self, contours)
        if mycnt != np.array([]):
            cv2.drawContours(procces, mycnt, -1, [255, 0, 0])
            Npoint = MyAlgorithm.getNpoint(self, mycontour=mycnt, N=3)
            if Npoint != np.array([]):
                procces = MyAlgorithm.drawNpoint(self, procces, Npoint, (255, 0, 0))

                self.actual_error, self.calc_actual_error, self.reference = MyAlgorithm.getError(self, Npoint)
                procces = MyAlgorithm.drawNpoint(self, procces, self.reference, (0, 0, 255))
                procces = MyAlgorithm.drawError(self, procces, reference=self.reference, points=Npoint)

                # First iteration
                if not self.calc_previous_error:
                    self.calc_previous_error = self.calc_actual_error

                state = MyAlgorithm.getState(self, self.actual_error)

                calc_error_actual, vel, w, calc_error_inc = MyAlgorithm.stateOfMachine(self, state, self.actual_error,
                                                                                       self.calc_actual_error,
                                                                                       self.calc_previous_error)

                # Drawing info
                status_view = MyAlgorithm.getStatusImage(self, calc_error_actual, self.actual_error, calc_error_inc,
                                                         state, vel, w)  #
                self.calc_previous_error = calc_error_actual
                self.setRightImageFiltered(status_view)

        # SHOW THE FILTERED IMAGE ON THE GUI
        self.setLeftImageFiltered(procces)
