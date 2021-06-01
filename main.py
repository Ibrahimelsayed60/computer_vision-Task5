from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt
from task import Ui_MainWindow
import sys
import math
import performance as pc
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.handle_ui()
        self.ui.pushButton_1.clicked.connect(self.draw_roc_curve_data)
        self.model()



    def handle_ui(self):
        self.ui.pushButton_1.setEnabled(False)
        

    def model(self):
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        #import pandas as pd
        #import numpy as np
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt

        X, y = make_classification(n_samples=1000, n_informative=10, n_features=20, flip_y=0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        prob_vector = model.predict_proba(X_test)[:, 1]
        prediction = model.predict(X_test)
        self.ui.pushButton_1.setEnabled(True)

        return prob_vector, y_test,prediction

    def draw_roc_curve_data(self):
        prob_vector,y_test,prediction = self.model()
        ROC = pc.roc_from_scratch(prob_vector,y_test,partitions=10)
        self.ui.image_1.plotItem.getViewBox().viewRange()
        self.ui.image_1.plot(ROC[:,0],ROC[:,1])

        





   




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()







    