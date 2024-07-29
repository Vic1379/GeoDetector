# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'optionsdialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DialogOptions(object):
    def setupUi(self, DialogOptions):
        DialogOptions.setObjectName("DialogOptions")
        DialogOptions.resize(580, 450)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DialogOptions.sizePolicy().hasHeightForWidth())
        DialogOptions.setSizePolicy(sizePolicy)
        DialogOptions.setModal(False)
        self.verticalLayout = QtWidgets.QVBoxLayout(DialogOptions)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_seg = QtWidgets.QLabel(DialogOptions)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_seg.setFont(font)
        self.label_seg.setAlignment(QtCore.Qt.AlignCenter)
        self.label_seg.setObjectName("label_seg")
        self.verticalLayout.addWidget(self.label_seg)
        self.cb_segSorted = QtWidgets.QCheckBox(DialogOptions)
        self.cb_segSorted.setChecked(True)
        self.cb_segSorted.setObjectName("cb_segSorted")
        self.verticalLayout.addWidget(self.cb_segSorted)
        self.cb_rawMasks = QtWidgets.QCheckBox(DialogOptions)
        self.cb_rawMasks.setObjectName("cb_rawMasks")
        self.verticalLayout.addWidget(self.cb_rawMasks)
        self.line = QtWidgets.QFrame(DialogOptions)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.btn_clsOpt = QtWidgets.QPushButton(DialogOptions)
        self.btn_clsOpt.setObjectName("btn_clsOpt")
        self.horizontalLayout.addWidget(self.btn_clsOpt)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(DialogOptions)
        QtCore.QMetaObject.connectSlotsByName(DialogOptions)

    def retranslateUi(self, DialogOptions):
        _translate = QtCore.QCoreApplication.translate
        DialogOptions.setWindowTitle(_translate("DialogOptions", "Параметры программы"))
        self.label_seg.setText(_translate("DialogOptions", "Сегментация"))
        self.cb_segSorted.setText(_translate("DialogOptions", "Сегментировать только отсортированные снимки"))
        self.cb_rawMasks.setText(_translate("DialogOptions", "Генерировать сырые маски"))
        self.btn_clsOpt.setText(_translate("DialogOptions", "Закрыть"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DialogOptions = QtWidgets.QDialog()
    ui = Ui_DialogOptions()
    ui.setupUi(DialogOptions)
    DialogOptions.show()
    sys.exit(app.exec_())
