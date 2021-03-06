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
        self.motors = motors
        self.imageRight = None
        self.imageLeft = None
        self.contVel = 0
        self.flag_time = True
        self.flag_go = True

        self.lap = 0
        self.flap_lap = True
        self.flag_no_line = False

        self.actual_error = np.array([0,0,0])
        self.calc_actual_error = 0
        self.calc_previous_error = 0
        self.calc_error_inc =0
        self.vel = 0
        self.w = 0

        self.ref3point = np.array(([316, 244], [235, 361], [153, 478]), dtype=np.int)
        self.m = float(self.ref3point[2, 1] - self.ref3point[0, 1]) / float(
            (self.ref3point[2, 0] - self.ref3point[0, 0]))
        self.n = self.ref3point[0, 1] - self.m * self.ref3point[0, 0]

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



    def  getNpoint(self, mycontour, N):

        self.min_p = np.min(mycontour[:, 0])
        ejeY = np.array([self.min_p, self.min_p+ 50, self.min_p+ 70])
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

        if actual_error[0] < -50  and self.state != "straight":
            self.state = "right"
            self.contVel = 0
        elif actual_error[0] > 50  and self.state != "straight":
            self.state = "left"
            self.contVel = 0
        elif -40<actual_error[0]<0 and (self.state == "right" or self.state == "out curveR"):
            self.state = "out curveR"
            self.contVel +=1
        elif 0<actual_error[0]<40 and (self.state == "left" or self.state == "out curveL"):
            self.state = "out curveL"
            self.contVel +=1

        elif (self.state != "right" and self.state != "left") and -15<actual_error[0] < 0 and abs(actual_error[2])-abs(actual_error[0]) > 80:
            self.state = "close straightR"
            self.contVel += 1

        elif ( self.state != "right" and self.state != "left") and -0<actual_error[0] < 15 and abs(actual_error[2])-abs(actual_error[0]) > 80:
            self.state = "close straightL"
            self.contVel +=1


        elif (actual_error[0] < -10) and (self.state == "straight" or self.state == "close curveR" or self.state == "close straightL" or self.state == "close straightR") and abs(actual_error[2])-abs(actual_error[0]) < 80:
            self.state = "close curveR"


        elif (actual_error[0] > 10) and (self.state == "straight" or self.state == "close curveL" or self.state == "close straightL" or self.state == "close straightR") and abs(actual_error[2])-abs(actual_error[0]) < 80:
            self.state = "close curveL"


        elif (-10<actual_error[0]<10) and self.state != "right" and self.state != "left":
            self.state = "straight"
            self.contVel +=1

        if self.contVel > 45:
            self.contVel = 45


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
        self.calc_error_inc = abs(calc_error_actual - calc_previous_error)
        velMax = 10.5

        # # Disacelerate
        if state == "close curveR":

            self.vel = (1 - float(abs(actual_error[0])) / 100) * 2 + 6
            Kwp = -0.00525
            Kwd = -0.0003
            self.w = Kwp * calc_error_actual + Kwd * self.calc_error_inc
        elif state == "close curveL":
            self.vel = (1 - float(abs(actual_error[0])) / 100) * 2 + 6
            Kwp = 0.00525
            Kwd = 0.0003
            self.w = Kwp * calc_error_actual + Kwd * self.calc_error_inc

        elif state == "straight":
            self.vel = velMax+0.1*self.contVel
            self.w = 0

        elif state == "right":
            Kvp = 0.007
            Kvd = 0.004
            Kwp = -0.00425
            Kwd = -0.004
            self.vel = 6 - Kvp * calc_error_actual - Kvd * self.calc_error_inc
            self.w = Kwp * calc_error_actual + Kwd * self.calc_error_inc
        elif state == "left":

            Kvp = 0.008
            Kvd = 0.004
            Kwp = 0.00425
            Kwd = 0.0045
            self.vel = 6 - Kvp * calc_error_actual - Kvd * self.calc_error_inc
            self.w = Kwp * calc_error_actual + Kwd * self.calc_error_inc

        elif state == "close straightR":
            self.vel = velMax+0.1*self.contVel
            Kwp = -0.00045
            Kwd = -0.0003
            self.w = Kwp * calc_error_actual + Kwd * self.calc_error_inc

        elif state == "close straightL":
            self.vel = velMax+0.1*self.contVel
            Kwp = 0.00045
            Kwd = 0.0003
            self.w = Kwp * calc_error_actual + Kwd * self.calc_error_inc

        elif state == "out curveR":
            self.vel = self.vel+0.1*self.contVel
            Kwp = -0.0006
            if self.contVel > 20:
                kwp = -0.0007
            Kwd = -0.00035
            if self.vel > 15:
                self.vel = 15
                self.w = Kwp * calc_error_actual + Kwd * self.calc_error_inc
        elif state == "out curveL":
            self.vel = self.vel+0.1*self.contVel

            Kwp = 0.0006
            if self.contVel > 20:
                kwp = 0.0007
            Kwd = 0.00035
            if self.vel > 15:
                self.vel = 15
                self.w = Kwp * calc_error_actual + Kwd * self.calc_error_inc

        elif state == "not line":
            if self.previous_state == "left" or self.previous_state == "close curveL" or self.previous_state == "out curveL":
                self.w = 0.5
                self.vel = -2.75
            else:
                self.w = -0.5
                self.vel = -2.75
         


        self.motors.sendV(self.vel)
        self.motors.sendW(self.w)


    def execute(self):
        # GETTING THE IMAGES
        imageRight = self.cameraR.getImage()


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

        # Detect a line
        if line.size != 0:

            # Drawing line
            procces[line[:, 0], line[:, 1], :] = (255, 255, 255)

            # Get the 3 points which we use to calculate the error
            Npoint = MyAlgorithm.getNpoint(self, mycontour=line, N=3)
            if self.min_p<400:
                # Detect a good line
                self.flag_no_line = False
            else:
                self.flag_no_line = True


            if Npoint != np.array([]):


                self.flag_no_line = False

                # Drawing the 3 points
                procces = MyAlgorithm.drawNpoint(self, procces, Npoint, (255, 0, 0))

                # Get the 3 projected points to the line reference and the error with 3 points line
                self.actual_error, self.calc_actual_error, self.reference = MyAlgorithm.getError(self, Npoint)

                # Drawing the reference points and the error
                procces = MyAlgorithm.drawNpoint(self, procces, self.reference, (0, 0, 255))
                procces = MyAlgorithm.drawError(self, procces, reference=self.reference, points=Npoint)

                # Get the state with actual error
                MyAlgorithm.getState(self, self.actual_error)


        else:
            self.flag_no_line = True

        # If not detect a good line
        if self.flag_no_line:
            if self.state != "not line":
                self.previous_state = self.state
            self.state = "not line"



        # After ge
        MyAlgorithm.stateOfMachine(self, self.state, self.actual_error,self.calc_actual_error,self.calc_previous_error)

        # Drawing info
        status_view = MyAlgorithm.getStatusImage(self, self.calc_actual_error, self.actual_error, self.calc_error_inc, self.state,
                                                 self.vel, self.w)

        self.setRightImageFiltered(status_view)
        self.calc_previous_error = self.calc_actual_error



        # SHOW THE FILTERED IMAGE ON THE GUI
        self.setLeftImageFiltered(procces)





