from gui.widgets.teleopWidget import TeleopWidget

__author__ = 'frivas'


import resources_rc
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow
from gui.form import Ui_MainWindow
from gui.widgets.cameraWidget import CameraWidget
from gui.widgets.logoWidget import LogoWidget



class MainWindow(QMainWindow, Ui_MainWindow):

    updGUI=pyqtSignal()
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.teleop=TeleopWidget(self)
        self.tlLayout.addWidget(self.teleop)
        self.teleop.setVisible(True)
        self.logo = LogoWidget(self)
        self.logoLayout.addWidget(self.logo)
        self.logo.setVisible(True)

        self.buffer = []



        self.pushButton.clicked.connect(self.playClicked)
        self.pushButton.setCheckable(True)
        self.updGUI.connect(self.updateGUI)
        self.camera1=CameraWidget(self)

        self.stopButton.clicked.connect(self.stopClicked)

    def updateGUI(self):
        #print 'update gui'
        self.camera1.updateImage()
        self.pose = self.pose_client.getPose3d()
        #self.sensorsWidget.sensorsUpdate.emit()


    def getCameraL(self):
        return self.cameraL

    def setPose(self,pose_client):
        self.pose_client = pose_client

    def setCameraL(self,camera):
        self.cameraL=camera

    def getCameraR(self):
        return self.cameraR

    def setCameraR(self,camera):
        self.cameraR=camera

    def getMotors(self):
        return self.motors

    def setMotors(self,motors):
        self.motors=motors

    def playClicked(self):
        if self.pushButton.isChecked():
            self.pushButton.setText('RUNNING')
            self.pushButton.setStyleSheet("background-color: green")
            self.algorithm.play()
        else:
            self.pushButton.setText('STOPPED')
            self.pushButton.setStyleSheet("background-color: red")
            self.algorithm.stop()

    def setAlgorithm(self, algorithm ):
        self.algorithm=algorithm

    def getAlgorithm(self):
        return self.algorithm

    def setXYValues(self,newX,newY):
        # print ("newX: %f, newY: %f" % (newX, newY) )
        write=False

        myW=-newX*self.motors.getMaxW()
        myV=-newY*self.motors.getMaxV()
        # print "myV: ",myV,"myW: ",myW
        if write:
            pose = self.pose_client.getPose3d()
            self.buffer.append("{:f} {:f} {:f} {:f}\n".format(pose.x/1000,pose.y/1000,myV,myW))
            if len(self.buffer)==10:
                with open("/home/f/PycharmProjects/Robotica/follow_line/laps/prueba.txt",'a') as file:
                    file.write("".join(self.buffer))
                self.buffer=[]


        self.motors.sendV(myV)
        self.motors.sendW(myW)

    def stopClicked(self):
        self.motors.sendV(0)
        self.motors.sendW(0)
        self.teleop.returnToOrigin()

    def closeEvent(self, event):
        self.algorithm.kill()
        self.cameraR.stop()
        self.cameraL.stop()
        event.accept()
