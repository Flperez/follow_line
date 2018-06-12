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
        self.contVel = 0
        self.flag_time = True
        self.flag_go = True

        self.lap = 0
        self.flap_lap = True

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



    def getNpoint(self, mycontour, N):

        min_p = np.min(mycontour[:, 0])
        ejeY = np.array([min_p, min_p+ 50, min_p+ 70])
        try:
            listmax = [0.5 *np.max(mycontour[np.where(mycontour[:, 0] == line),1]) for line in ejeY]
            listmin = [0.5 *np.min(mycontour[np.where(mycontour[:, 0] == line),1]) for line in ejeY]
            ejeX =  (np.array(listmax,dtype=int)+np.array(listmin,dtype=int))
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

        if actual_error[0] < -50 and actual_error[2] > 50 and self.state != "straight":
            self.state = "right"
            self.contVel = 0
        elif actual_error[0] > 50 and actual_error[2] < -50  and self.state != "straight":
            self.state = "left"
            self.contVel = 0
        elif actual_error[0] < -50:
            self.state = "right"
            self.contVel = 0
        elif actual_error[0] > 50:
            self.state = "left"
            self.contVel = 0

        elif (self.state != "right" and self.state != "left") and -30<actual_error[0] < 0 and abs(actual_error[2])-abs(actual_error[0]) > 80:
            self.state = "out curveR"
            self.contVel += 1

        elif ( self.state != "right" and self.state != "left") and -0<actual_error[0] < 30 and abs(actual_error[2])-abs(actual_error[0]) > 80:
            self.state = "out curveL"
            self.contVel +=1

        elif (actual_error[0] < -10) and (self.state == "straight" or self.state == "close curveR" or self.state == "out curveL" or self.state == "out curveR") and abs(actual_error[2])-abs(actual_error[0]) < 100:
            self.state = "close curveR"


        elif (actual_error[0] > 10) and (self.state == "straight" or self.state == "close curveL" or self.state == "out curveL" or self.state == "out curveR") and abs(actual_error[2])-abs(actual_error[0]) < 100:
            self.state = "close curveL"
        elif (-10<actual_error[0]<10):
            self.state = "straight"
            self.contVel +=1

        if self.contVel > 45:
            self.contVel = 45

        return self.state

    def calcError(self, error, alfa, beta, gamma):
        return alfa * abs(error[0]) + beta * abs(error[1]) + gamma * abs(error[2])

    def getStatusImage(self, calc_error_actual, actual_error, calc_error_inc, state, vel, w):
        out = np.zeros([480, 640, 3], dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, "State: %s" % (state), (10, 150), font, 1, (255, 255, 255))
        cv2.putText(out, "v: %f   w: %f" % (vel, w), (10, 200), font, 1, (255, 0, 255))
        cv2.putText(out, "CalcErr: %2.f" % (calc_error_actual), (10, 250), font, 1, (255, 255, 0))
        cv2.putText(out,"TIME: %02d:%02d"%(divmod(int(time.clock()-self.time_go),60)),(100, 350), font, 2, (0, 255, 0),2)
        # Close to line lap
        if  self.initial_pose.x/1000 - 0.5 < self.pose.x/1000 < self.initial_pose.x/1000 + 0.5 \
                and self.initial_pose.y/1000  < self.pose.y/1000 < self.initial_pose.y/1000+2:
            # Not increase various time
            if self.flap_lap:
                self.lap+=1
                self.flap_lap = False
        else:

            self.flap_lap = True
        print self.lap


        # if self.pose.
        #cv2.putText(out, "Err: %2.f %2.f %2.f" % (actual_error[0], actual_error[1], actual_error[2]), (10, 300), font,
        #            1, (255, 255, 0))
        #cv2.putText(out, "inc: %2.f" % (calc_error_inc), (10, 350), font, 1, (255, 255, 0))
        return out

    def stateOfMachine(self, state, actual_error, calc_error_actual, calc_previous_error):
        calc_error_inc = abs(calc_error_actual - calc_previous_error)
        # print calc_error_actual - calc_previous_error
        # print self.contVel
        velMax = 10.5

        # # Disacelerate
        if state == "close curveR":

            vel = (1 - float(abs(actual_error[0])) / 100) * 2 + 7.5
            Kwp = -0.00525
            Kwd = -0.0003
            w = Kwp * calc_error_actual + Kwd * calc_error_inc
        elif state == "close curveL":
            vel = (1 - float(abs(actual_error[0])) / 100) * 2 + 7.5
            Kwp = 0.000525
            Kwd = 0.0003
            w = Kwp * calc_error_actual + Kwd * calc_error_inc

        elif state == "straight":
            vel = velMax+0.1*self.contVel
            w = 0

        elif state == "right" or state == "strong right":
            Kvp = 0.005
            Kvd = 0.004
            Kwp = -0.0055
            Kwd = -0.004
            vel = 6.5 - Kvp * calc_error_actual - Kvd * calc_error_inc
            w = Kwp * calc_error_actual + Kwd * calc_error_inc
        elif state == "left" or state == "strong left":

            Kvp = 0.0054
            Kvd = 0.004
            Kwp = 0.0059
            Kwd = 0.0045
            vel = 6.5 - Kvp * calc_error_actual - Kvd * calc_error_inc

            w = Kwp * calc_error_actual + Kwd * calc_error_inc
        elif state == "out curveR":
            vel = velMax+0.08*self.contVel
            Kwp = -0.00045
            Kwd = -0.0003
            w = Kwp * calc_error_actual + Kwd * calc_error_inc

        elif state == "out curveL":
            vel = velMax+0.08*self.contVel
            Kwp = 0.00045
            Kwd = 0.0003
            w = Kwp * calc_error_actual + Kwd * calc_error_inc
        elif state == "not line":
            w = -0.3
            vel = 0

        self.motors.sendV(vel)
        self.motors.sendW(w)
        return calc_error_actual, vel, w, calc_error_inc

    def execute(self):
        # GETTING THE IMAGES
        if self.flag_go:
            self.time_go = time.clock()
            self.initial_pose = self.pose_client.getPose3d()
            self.flag_go = False


        imageRight = self.cameraR.getImage()
        self.pose = self.pose_client.getPose3d()

        # Limitamos la imagen de entrada
        imageRight = imageRight.data
        imageRightdata = np.zeros([480, 640, 3], dtype=np.uint8)
        procces = np.zeros([480, 640, 3], dtype=np.uint8)
        imageRightdata[480 / 2:, :] = imageRight[480 / 2:, :]

        # Mascara para la linea
        mask = MyAlgorithm.createmask(self, imageRightdata)

        # Extraemos el contorno de la linea
        line = np.asarray(np.where(mask==255)).astype(np.int).T
        procces[line[:,0],line[:,1],:]=(255,0,255)
        if line.size != 0:

            # Drawing line
            procces[line[:, 0], line[:, 1], :] = (255, 255, 255)

            # Get the 3 points which we use to calculate the error
            Npoint = MyAlgorithm.getNpoint(self, mycontour=line, N=3)
            if Npoint != np.array([]):

                # Drawing the 3 points
                procces = MyAlgorithm.drawNpoint(self, procces, Npoint, (255, 0, 0))

                # Get the 3 projected points to the line reference and the error with 3 points line
                self.actual_error, self.calc_actual_error, self.reference = MyAlgorithm.getError(self, Npoint)

                # Drawing the reference points and the error
                procces = MyAlgorithm.drawNpoint(self, procces, self.reference, (0, 0, 255))
                procces = MyAlgorithm.drawError(self, procces, reference=self.reference, points=Npoint)

                # First iteration
                if not self.calc_previous_error:
                    self.calc_previous_error = self.calc_actual_error

                # Get the state with actual error
                state = MyAlgorithm.getState(self, self.actual_error)
                calc_error_actual, vel, w, calc_error_inc = MyAlgorithm.stateOfMachine(self, state, self.actual_error,
                                                                                       self.calc_actual_error,
                                                                                       self.calc_previous_error)

                # Drawing info
                status_view = MyAlgorithm.getStatusImage(self, calc_error_actual, self.actual_error, calc_error_inc, state, vel, w)  #
                self.calc_previous_error = calc_error_actual
                self.setRightImageFiltered(status_view)
        else:
            # Add your code here
            if self.flag_time:
                self.time_not_line = time.clock()
                self.flag_time = False
            else:
                # print  time.clock() -self.time_not_line
                if  time.clock() - self.time_not_line>0.5:
                    self.state = "not line"

        # SHOW THE FILTERED IMAGE ON THE GUI
        self.setLeftImageFiltered(procces)
