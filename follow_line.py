#!/usr/bin/python3
#
#  Copyright (C) 1997-2016 JDE Developers Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/.
#  Authors :
#       Aitor Martinez Fernandez <aitor.martinez.fernandez@gmail.com>
#       Francisco Miguel Rivas Montero <franciscomiguel.rivas@urjc.es>
#


import sys
import config
import comm
from gui.GUI import MainWindow
from gui.threadGUI import ThreadGUI
from parallelIce.cameraClient import CameraClient
from parallelIce.motors import Motors
import easyiceconfig as EasyIce
from MyAlgorithm import MyAlgorithm
from PyQt5.QtWidgets import QApplication



if __name__ == "__main__":

    cfg = config.load(sys.argv[1])
    #starting comm
    jdrc= comm.init(cfg, 'FollowLineF1')

    cameraL = jdrc.getCameraClient("FollowLineF1.CameraLeft")
    cameraR = jdrc.getCameraClient("FollowLineF1.CameraRight")
    motors = jdrc.getMotorsClient("FollowLineF1.Motors")
    pose_client = jdrc.getPose3dClient("FollowLineF1.Pose")
    algorithm=MyAlgorithm(cameraL, cameraR, motors, pose_client)

    app = QApplication(sys.argv)
    myGUI = MainWindow()
    myGUI.setCameraL(cameraL)
    myGUI.setPose(pose_client)
    myGUI.setCameraR(cameraR)
    myGUI.setMotors(motors)
    myGUI.setAlgorithm(algorithm)
    myGUI.show()


    t2 = ThreadGUI(myGUI)
    t2.daemon=True
    t2.start()


    sys.exit(app.exec_())
