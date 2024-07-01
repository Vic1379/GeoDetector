import sys, random, os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import numpy as np, torch as trc, matplotlib.pyplot as plt
import rasterio
from osgeo import gdal

from app_base_UI import Ui_MainWindow

PATH = os.path.dirname(os.path.abspath(__file__))

def open_SRC():
    '''filePath, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Open Image', 'Desktop',
        'PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*)')'''
    filePath = QtWidgets.QFileDialog.getExistingDirectory(window, 'Выберите директорию для сканирования', 'Desktop')
    if filePath:
        ui.line_strSrc.setText(filePath)

def scan_SRC():
    path, files = ui.line_strSrc.text(), []
    if os.path.isdir(path):
        model.removeRows(0, model.rowCount())
        # ui.list_foundImgs.clear()
        if ui.cb_scanSubs.isChecked():
            for content in os.walk(path):
                for file in content[2]:
                    if file[-11:] == '.PAN.L2.tif':
                        files.append(os.path.join(content[0], file))
        else:
            for file in os.listdir(path):
                if file[-11:] == '.PAN.L2.tif':
                    files.append(os.path.join(path, file))
        
        if len(files) == 0:
            QtWidgets.QMessageBox.information(window, 'Результаты сканирования', 'Файлы не найдены...           ')
        else:
            for file in files:
                item = QtGui.QStandardItem(file)
                item.setCheckable(True)
                item.setEditable(False)
                item.setCheckState(2)
                # item.clicked.connect(lambda: show_IMG(item.text()))
                model.appendRow(item)
            
            if str(len(files))[-1] == '1':
                QtWidgets.QMessageBox.information(window, 'Результаты сканирования', 'В результате сканирования был найден '+str(len(files))+' файл.')
            elif str(len(files))[-1] in ('2', '3', '4'):
                QtWidgets.QMessageBox.information(window, 'Результаты сканирования', 'В результате сканирования было найдено '+str(len(files))+' файла.')
            else:
                QtWidgets.QMessageBox.information(window, 'Результаты сканирования', 'В результате сканирования было найдено '+str(len(files))+' файлов.')
    else:
        QtWidgets.QMessageBox.warning(window, 'Ошибка сканирования', 'Такой директории не существует...')

def show_IMG():
    # add scaling
    image = QtGui.QImage(ui.list_foundImgs.selectedIndexes()[0].data())

    vp_w = ui.graphicsView.width()-30
    if image.width() > vp_w:
        pixmap = QtGui.QPixmap.fromImage(image).scaled(vp_w, vp_w)
    else:
        pixmap = QtGui.QPixmap.fromImage(image)
    
    ui.graphicsView.setScene(QtWidgets.QGraphicsScene())
    ui.graphicsView.scene().addPixmap(pixmap)

def set_OUT():
    filePath = QtWidgets.QFileDialog.getExistingDirectory(window, 'Выберите директорию для записи результатов нарезки', 'Desktop')
    if filePath:
        ui.line_strWrite.setText(filePath)

def cut_imgs():
    path_base = ui.line_strWrite.text()
    if os.path.isdir(path_base):
        model, items = ui.list_foundImgs.model(), []
        for i in range(model.rowCount()):
            item = model.item(i)
            if item.checkState() == 2: items.append(item)
        if len(items) > 0:
            h, w, cut_ind = ui.sb_height.value(), ui.sb_width.value(), 0
            for item in items:
                image = rasterio.open(item.text())
                tf, crs = image.meta['transform'], image.crs.to_string()
                top_cords = (image.bounds[0], image.bounds[3])
                img_data = np.squeeze(image.read())

                # print(img_data.shape)
                height, width = img_data.shape
                h_ost, w_ost = height%h, width%w
                h_str, w_str = h_ost//2, w_ost//2

                # print(h_ost, w_ost)
                print(h_str, w_str)
                if h_ost != 0:
                    img_data = img_data[h_str:-h_ost//2]
                    # print(h_ost//2, -h_ost//2)
                if w_ost != 0:
                    img_data = img_data[:, w_str:-w_ost//2]
                    # print(w_ost//2, -w_ost//2)
                # print(img_data.shape)

                cuts = []
                for i in range(img_data.shape[0]//h):
                    for j in range(img_data.shape[1]//w):
                        tf_local = list(tf)
                        tf_local[2] = top_cords[0] + (j*w+w_str)*tf[0]+1.05
                        tf_local[5] = top_cords[1] - (i*h+h_str)*tf[0]-1.05
                        
                        cut = img_data[i*h:(i+1)*h, j*w:(j+1)*w]
                        if ui.cb_filter.isChecked():
                            if (cut==0).sum()/(h*w) <= 1-ui.sb_filter.value()/100:
                                cuts.append((cut, crs, tf_local))
                        else:
                            cuts.append((cut, crs, tf_local))
                image.close()

                if ui.cb_group.isChecked():
                    path = os.path.join(path_base, item.text().split('\\')[-1])
                    if not os.path.isdir(path): os.mkdir(path)
                    cut_ind = 0
                else: path = path_base
                print(path)
                
                for cut in cuts:
                    output = os.path.join(path, str(cut_ind)+'_cut.tif')
                    x_pixels, y_pixels = cut[0].shape

                    driver = gdal.GetDriverByName('GTiff')
                    dataset = driver.Create(output, x_pixels, y_pixels, 1, gdal.GDT_Float32)
                    # dataset = driver.Create(output, x_pixels, y_pixels, 1, gdal.GDT_Int32)
                    dataset.GetRasterBand(1).WriteArray(cut[0]/255)
                    # dataset.GetRasterBand(1).WriteArray(cut[0])
                    dataset.FlushCache()
                    dataset=None

                    output_dt = os.path.join(path, str(cut_ind)+'_cut.tfw')
                    with open(output_dt, 'w') as file:
                        file.write(str(cut[2][0])+'\n')
                        file.write('0\n')
                        file.write('0\n')
                        file.write(str(cut[2][4])+'\n')
                        file.write(str(cut[2][2])+'\n')
                        file.write(str(cut[2][5])+'\n')
                    
                    cut_ind += 1
        else:
            QtWidgets.QMessageBox.warning(window, 'Ошибка записи', 'Снимки не выбраны...')
    else:
        QtWidgets.QMessageBox.warning(window, 'Ошибка записи', 'Такой директории не существует...')

app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(window)

model = QtGui.QStandardItemModel()
ui.list_foundImgs.setModel(model)
# ui.list_foundImgs.clicked.connect(show_IMG)
# model.itemChanged.connect(show_IMG)
ui.list_foundImgs.selectionModel().selectionChanged.connect(show_IMG)

preview_scene = QtWidgets.QGraphicsScene()
ui.graphicsView.setInteractive(True)
ui.graphicsView.setScene(preview_scene)
ui.graphicsView.setDragMode(1)
# print(ui.graphicsView.width())

ui.btn_findSrc.clicked.connect(open_SRC)
ui.btn_startScan.clicked.connect(scan_SRC)
ui.btn_findWloc.clicked.connect(set_OUT)
ui.btn_start.clicked.connect(cut_imgs)

window.show()
sys.exit(app.exec_())
