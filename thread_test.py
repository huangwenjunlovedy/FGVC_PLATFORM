# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import time

class MainWidget(QWidget):
    def __init__(self, parent =None):
        super(MainWidget, self).__init__(parent)
        self.setWindowTitle("QThread 例子")
        # self.thread = Worker()
        self.textBrowser = self.textBrowser = QTextBrowser()
        self.listFile = QListWidget()
        self.btnStart = QPushButton("开始")
        layout = QGridLayout(self)
        layout.addWidget(self.listFile,0,0,1,2)
        layout.addWidget(self.btnStart,1,1)
        self.btnStart.clicked.connect(self.slotStart)
        # self.thread.sinout.connect(self.slotAdd)
        # 实时显示输出, 将控制台的输出重定向到界面中
        sys.stdout = Signal()
        sys.stdout.text_update.connect(self.updatetext)

    def slotStart(self):
        for i in range(5):
            print('i=', i)
            time.sleep(1)
    print('sys.stdout', sys.stdout)

    def updatetext(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.textBrowser.append(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()


class Signal(QObject):
    text_update = pyqtSignal(str)
    def write(self, text):
        self.text_update.emit(str(text))
        QApplication.processEvents()

# class Worker(QThread):
#     sinout=  pyqtSignal(str)
#     def __init__(self, parent = None):
#         super(Worker,self).__init__(parent)
#         self.working = True
#         self.num = 0
#
#     def __del__(self):
#         self.working = False
#         self.wait()
#
#     def run(self):
#         while self.working == True:
#             file_str = "File index {0} ".format(self.num)
#             self.num += 1
#             # 发射信号
#             self.sinout.emit(file_str)
#             # 线程休眠2秒
#             self.sleep(2)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainWidget()
    demo.show()
    sys.exit(app.exec_())
