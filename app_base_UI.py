# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'geodetector.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setStyleSheet("QTabBar::tab{height: 40px; width:300px; color: black; font: 17pt;}\n"
"QTabWidget::tab-bar{alignment: center;}")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_start = QtWidgets.QWidget()
        self.tab_start.setObjectName("tab_start")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.tab_start)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.vLayout_browse = QtWidgets.QVBoxLayout()
        self.vLayout_browse.setObjectName("vLayout_browse")
        self.label_chooseDir = QtWidgets.QLabel(self.tab_start)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_chooseDir.setFont(font)
        self.label_chooseDir.setStyleSheet("")
        self.label_chooseDir.setAlignment(QtCore.Qt.AlignCenter)
        self.label_chooseDir.setObjectName("label_chooseDir")
        self.vLayout_browse.addWidget(self.label_chooseDir)
        self.hLayout_findSrc = QtWidgets.QHBoxLayout()
        self.hLayout_findSrc.setObjectName("hLayout_findSrc")
        self.line_strSrc = QtWidgets.QLineEdit(self.tab_start)
        self.line_strSrc.setObjectName("line_strSrc")
        self.hLayout_findSrc.addWidget(self.line_strSrc)
        self.btn_findSrc = QtWidgets.QPushButton(self.tab_start)
        self.btn_findSrc.setObjectName("btn_findSrc")
        self.hLayout_findSrc.addWidget(self.btn_findSrc)
        self.vLayout_browse.addLayout(self.hLayout_findSrc)
        self.cb_scanSubs = QtWidgets.QCheckBox(self.tab_start)
        self.cb_scanSubs.setChecked(True)
        self.cb_scanSubs.setObjectName("cb_scanSubs")
        self.vLayout_browse.addWidget(self.cb_scanSubs)
        self.btn_startScan = QtWidgets.QPushButton(self.tab_start)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.btn_startScan.setFont(font)
        self.btn_startScan.setObjectName("btn_startScan")
        self.vLayout_browse.addWidget(self.btn_startScan)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.vLayout_browse.addItem(spacerItem)
        self.label_foundImgs = QtWidgets.QLabel(self.tab_start)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_foundImgs.setFont(font)
        self.label_foundImgs.setStyleSheet("")
        self.label_foundImgs.setAlignment(QtCore.Qt.AlignCenter)
        self.label_foundImgs.setObjectName("label_foundImgs")
        self.vLayout_browse.addWidget(self.label_foundImgs)
        self.list_foundImgs = QtWidgets.QListView(self.tab_start)
        self.list_foundImgs.setObjectName("list_foundImgs")
        self.vLayout_browse.addWidget(self.list_foundImgs)
        self.horizontalLayout_3.addLayout(self.vLayout_browse)
        self.vLayout_prevANDproc = QtWidgets.QVBoxLayout()
        self.vLayout_prevANDproc.setObjectName("vLayout_prevANDproc")
        self.label_preview = QtWidgets.QLabel(self.tab_start)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_preview.setFont(font)
        self.label_preview.setStyleSheet("")
        self.label_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.label_preview.setObjectName("label_preview")
        self.vLayout_prevANDproc.addWidget(self.label_preview)
        self.graphicsView = QtWidgets.QGraphicsView(self.tab_start)
        self.graphicsView.setObjectName("graphicsView")
        self.vLayout_prevANDproc.addWidget(self.graphicsView)
        self.hLayout_outopt = QtWidgets.QHBoxLayout()
        self.hLayout_outopt.setObjectName("hLayout_outopt")
        self.vLayout_write = QtWidgets.QVBoxLayout()
        self.vLayout_write.setObjectName("vLayout_write")
        self.label = QtWidgets.QLabel(self.tab_start)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.vLayout_write.addWidget(self.label)
        self.hLayout_writeLoc = QtWidgets.QHBoxLayout()
        self.hLayout_writeLoc.setObjectName("hLayout_writeLoc")
        self.line_strWrite = QtWidgets.QLineEdit(self.tab_start)
        self.line_strWrite.setObjectName("line_strWrite")
        self.hLayout_writeLoc.addWidget(self.line_strWrite)
        self.btn_findWloc = QtWidgets.QPushButton(self.tab_start)
        self.btn_findWloc.setObjectName("btn_findWloc")
        self.hLayout_writeLoc.addWidget(self.btn_findWloc)
        self.vLayout_write.addLayout(self.hLayout_writeLoc)
        self.hLayout_outopt.addLayout(self.vLayout_write)
        self.line_2 = QtWidgets.QFrame(self.tab_start)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.hLayout_outopt.addWidget(self.line_2)
        self.vLayout_opt = QtWidgets.QVBoxLayout()
        self.vLayout_opt.setObjectName("vLayout_opt")
        self.label_2 = QtWidgets.QLabel(self.tab_start)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_2.setObjectName("label_2")
        self.vLayout_opt.addWidget(self.label_2)
        self.hLayout_opt = QtWidgets.QHBoxLayout()
        self.hLayout_opt.setObjectName("hLayout_opt")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_width = QtWidgets.QLabel(self.tab_start)
        self.label_width.setObjectName("label_width")
        self.horizontalLayout_2.addWidget(self.label_width)
        self.sb_width = QtWidgets.QSpinBox(self.tab_start)
        self.sb_width.setMinimum(1)
        self.sb_width.setMaximum(10000)
        self.sb_width.setProperty("value", 1280)
        self.sb_width.setObjectName("sb_width")
        self.horizontalLayout_2.addWidget(self.sb_width)
        self.label_height = QtWidgets.QLabel(self.tab_start)
        self.label_height.setObjectName("label_height")
        self.horizontalLayout_2.addWidget(self.label_height)
        self.sb_height = QtWidgets.QSpinBox(self.tab_start)
        self.sb_height.setMinimum(1)
        self.sb_height.setMaximum(10000)
        self.sb_height.setProperty("value", 1280)
        self.sb_height.setObjectName("sb_height")
        self.horizontalLayout_2.addWidget(self.sb_height)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.sb_filter = QtWidgets.QSpinBox(self.tab_start)
        self.sb_filter.setMaximum(100)
        self.sb_filter.setProperty("value", 50)
        self.sb_filter.setObjectName("sb_filter")
        self.horizontalLayout_4.addWidget(self.sb_filter)
        self.label_percent = QtWidgets.QLabel(self.tab_start)
        self.label_percent.setObjectName("label_percent")
        self.horizontalLayout_4.addWidget(self.label_percent)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.hLayout_opt.addLayout(self.verticalLayout_3)
        self.line = QtWidgets.QFrame(self.tab_start)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.hLayout_opt.addWidget(self.line)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.cb_group = QtWidgets.QCheckBox(self.tab_start)
        self.cb_group.setChecked(True)
        self.cb_group.setObjectName("cb_group")
        self.verticalLayout_2.addWidget(self.cb_group)
        self.cb_filter = QtWidgets.QCheckBox(self.tab_start)
        self.cb_filter.setChecked(True)
        self.cb_filter.setObjectName("cb_filter")
        self.verticalLayout_2.addWidget(self.cb_filter)
        self.hLayout_opt.addLayout(self.verticalLayout_2)
        self.vLayout_opt.addLayout(self.hLayout_opt)
        self.hLayout_outopt.addLayout(self.vLayout_opt)
        self.vLayout_prevANDproc.addLayout(self.hLayout_outopt)
        self.btn_start = QtWidgets.QPushButton(self.tab_start)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.btn_start.setFont(font)
        self.btn_start.setObjectName("btn_start")
        self.vLayout_prevANDproc.addWidget(self.btn_start)
        self.horizontalLayout_3.addLayout(self.vLayout_prevANDproc)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 3)
        self.tabWidget.addTab(self.tab_start, "")
        self.tab_net = QtWidgets.QWidget()
        self.tab_net.setObjectName("tab_net")
        self.tabWidget.addTab(self.tab_net, "")
        self.tab_geo = QtWidgets.QWidget()
        self.tab_geo.setObjectName("tab_geo")
        self.tabWidget.addTab(self.tab_geo, "")
        self.verticalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 27))
        self.menubar.setObjectName("menubar")
        self.menu_main = QtWidgets.QMenu(self.menubar)
        self.menu_main.setObjectName("menu_main")
        self.menu_opti = QtWidgets.QMenu(self.menubar)
        self.menu_opti.setObjectName("menu_opti")
        self.menu_info = QtWidgets.QMenu(self.menubar)
        self.menu_info.setObjectName("menu_info")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_open = QtWidgets.QAction(MainWindow)
        self.action_open.setObjectName("action_open")
        self.action_info = QtWidgets.QAction(MainWindow)
        self.action_info.setObjectName("action_info")
        self.action_exit = QtWidgets.QAction(MainWindow)
        self.action_exit.setObjectName("action_exit")
        self.menu_main.addAction(self.action_open)
        self.menu_main.addSeparator()
        self.menu_main.addAction(self.action_exit)
        self.menubar.addAction(self.menu_main.menuAction())
        self.menubar.addAction(self.menu_opti.menuAction())
        self.menubar.addAction(self.menu_info.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "GeoDetector"))
        self.label_chooseDir.setText(_translate("MainWindow", "Расположение исходных файлов"))
        self.btn_findSrc.setText(_translate("MainWindow", "Выбрать путь"))
        self.cb_scanSubs.setText(_translate("MainWindow", "Сканировать внутренние директории"))
        self.btn_startScan.setText(_translate("MainWindow", "Начать сканирование"))
        self.label_foundImgs.setText(_translate("MainWindow", "Найденные снимки"))
        self.label_preview.setText(_translate("MainWindow", "Предпросмотр"))
        self.label.setText(_translate("MainWindow", "Место записи результатов"))
        self.btn_findWloc.setText(_translate("MainWindow", "Выбрать путь"))
        self.label_2.setText(_translate("MainWindow", "Опции нарезки"))
        self.label_width.setText(_translate("MainWindow", "Ширина (px)"))
        self.label_height.setText(_translate("MainWindow", "Высота (px)"))
        self.label_percent.setText(_translate("MainWindow", "Минимальный процент пикселей снимка"))
        self.cb_group.setText(_translate("MainWindow", "Группировать фрагменты"))
        self.cb_filter.setText(_translate("MainWindow", "Фильтр пустых значений"))
        self.btn_start.setText(_translate("MainWindow", "Начать обработку"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_start), _translate("MainWindow", "Нарезка снимков"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_net), _translate("MainWindow", "Обработка нейросетью"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_geo), _translate("MainWindow", "Проверка участка"))
        self.menu_main.setTitle(_translate("MainWindow", "Меню"))
        self.menu_opti.setTitle(_translate("MainWindow", "Настройки"))
        self.menu_info.setTitle(_translate("MainWindow", "Справка"))
        self.action_open.setText(_translate("MainWindow", "Открыть"))
        self.action_info.setText(_translate("MainWindow", "Справка"))
        self.action_exit.setText(_translate("MainWindow", "Выход"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())