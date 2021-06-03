from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog,QDialog
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt
from task import Ui_MainWindow
import cv2
import sys
import math
import performance as pc
import detection as dt
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.handle_ui()
        self.ui.pushButton_1.clicked.connect(self.draw_roc_curve_data)
        
        self.ui.pushButton.clicked.connect(self.open_dialog_box)


    def handle_ui(self):
        self.ui.pushButton_1.setEnabled(False)
        

    

    def draw_roc_curve_data(self):
        prob_vector,y_test,prediction = 0 #self.model()
        ROC = pc.roc_from_scratch(prob_vector,y_test,partitions=10)
        self.ui.image_1.plotItem.getViewBox().viewRange()

        self.ui.textEdit_1.setPlaceholderText("{}%".format(pc.accuracy_metric(y_test,prediction)))

        self.ui.image_1.plot(ROC[:,0],ROC[:,1])
   
    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName(self,'open File','c\\', 'image files(*.jpg)')
        path = filename[0]
        pixmap =QPixmap(path)
        self.ui.label_3.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())
        dt.facedetection(cv2.imread(path))

    def showDetected(self):
        self.ui.label_4.setPixmap(QPixmap('img.png'))

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()







    