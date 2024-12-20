import sys, os, shutil, json, time, datetime
from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np, torch as trc, rasterio
from osgeo import gdal

from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from app_ui_main import Ui_MainWindow
from app_ui_opt import Ui_DialogOptions
from app_ui_sub import PopUpProgressBar

from app_nns import ClsNet, SegNet, Encoder, Decoder, ConvBlock, processDataset

ROOT_H = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(os.path.normpath(ROOT_H)) == '_internal': ROOT = os.path.dirname(ROOT_H)
else: ROOT = ROOT_H

THD_CUT, THD_NNS = QtCore.QThread(), QtCore.QThread()

NET_CLS = os.path.join(ROOT, 'MODEL_CLS')
NET_SEG = os.path.join(ROOT, 'MODEL_SEG')
SETTINGS = os.path.join(ROOT_H, 'settings.json')
SIZE, SHARPNESS = 256, 10

class Worker_CUT(QtCore.QObject):
	finished, aborted, progress = QtCore.pyqtSignal(str), QtCore.pyqtSignal(), QtCore.pyqtSignal(int)
	def __init__(self, imgPaths, path_base, thread):
		super().__init__()
		self.imgPaths, self.path_base = imgPaths, path_base
		self.thread, self.abort_flag = thread, False
	
	@QtCore.pyqtSlot()
	def run(self):
		start_time = time.time()
		self.progress.emit(1)
		h, w = ui.sb_height.value(), ui.sb_width.value()
		for p, imgPath in enumerate(self.imgPaths):
			image = rasterio.open(imgPath)
			tf, crs = image.meta['transform'], image.crs.to_string()
			top_cords = (image.bounds[0], image.bounds[3])
			img_data = np.squeeze(image.read())

			# print(img_data.shape)
			if len(img_data.shape) == 3:
				chn, height, width = img_data.shape
				img_data = img_data[:3].transpose(1, 2, 0)
			else:
				height, width = img_data.shape
			
			h_ost, w_ost = height%h, width%w
			if h_ost != 0:
				img_data = img_data[h_ost//2:-h_ost//2]
			if w_ost != 0:
				img_data = img_data[:, w_ost//2:-w_ost//2]
			# print(img_data.shape)
			# print(list(tf))

			cuts = []
			for i in range(height//h):
				for j in range(width//w):
					tf_local = list(tf)
					tf_local[2] = top_cords[0] + (j*w+w_ost//2)*tf[0] + tf[0]/2
					tf_local[5] = top_cords[1] - (i*h+h_ost//2)*tf[0] - tf[0]/2
					
					cut = img_data[i*h:(i+1)*h, j*w:(j+1)*w]
					if ui.cb_filter.isChecked():
						if (cut==0).sum()/(h*w) <= 1-ui.sb_filter.value()/100:
							cuts.append((cut, crs, tf_local))
					else:
						cuts.append((cut, crs, tf_local))
			image.close()

			if ui.cb_group.isChecked():
				path = os.path.join(self.path_base, imgPath.split('\\')[-1])
				if not os.path.isdir(path): os.mkdir(path)
			else: path = path_base
			
			for i, cut in enumerate(cuts):
				output = os.path.join(path, str(i)+'_cut.jpg')

				img = Image.fromarray(cut[0])
				img.save(output)
				'''driver = gdal.GetDriverByName('GTiff')
				dataset = driver.Create(output, w, h, 1, gdal.GDT_Float32)
				dataset.GetRasterBand(1).WriteArray(cut[0]/255)
				dataset.FlushCache()
				dataset = None'''

				output_dt = os.path.join(path, str(i)+'_cut.jgw')
				with open(output_dt, 'w') as file:
					file.write(str(cut[2][0])+'\n')
					file.write('0\n')
					file.write('0\n')
					file.write(str(cut[2][4])+'\n')
					file.write(str(cut[2][2])+'\n')
					file.write(str(cut[2][5])+'\n')
			
			app.processEvents()
			if self.abort_flag:
				self.aborted.emit()
				self.thread.quit()
				return
			self.progress.emit(int(100/len(self.imgPaths)*(p+1)))
		
		print('Done!')
		self.progress.emit(100)
		self.finished.emit(str(datetime.timedelta(seconds=int(time.time()-start_time))))
		self.thread.quit()
	
	def abort(self):
		self.abort_flag = True

class Worker_NNS(QtCore.QObject):
	finished, aborted, progress = QtCore.pyqtSignal(str), QtCore.pyqtSignal(), QtCore.pyqtSignal(int)
	def __init__(self, imgPaths, path_base, thread):
		super().__init__()
		self.imgPaths, self.path_base = imgPaths, path_base
		self.thread, self.abort_flag = thread, False

	@QtCore.pyqtSlot()
	def run(self):
		start_time = time.time()
		self.progress.emit(1)
		read_img_trfs = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((SIZE, SIZE)),
			transforms.ToTensor(),

			transforms.Grayscale(),
			transforms.RandomAdjustSharpness(SHARPNESS, 1),
		])
		n, k, max_prog_before_seg = len(self.imgPaths), 1, 5
		if ui.cb_CLS.isChecked() and ui.cb_SEG.isChecked():
			k = 2
		
		app.processEvents()
		if self.abort_flag:
			self.aborted.emit()
			self.thread.quit()
			return
		self.progress.emit(5)

		sorted_img_paths = []
		if ui.cb_CLS.isChecked():
			out = os.path.join(self.path_base, 'CLS')
			out_txt = os.path.join(out, 'output.txt')

			dt_process = processDataset(self.imgPaths, read_img_trfs)
			batch_size = min(128, len(dt_process))
			process_dl = trc.utils.data.DataLoader(dt_process, batch_size)

			n_steps = n // batch_size
			if n % batch_size != 0:
				n_steps += 1
			
			net = trc.load(NET_CLS)
			preds, imgNames, imgPaths = trc.tensor([]), [], []
			for i, batch in enumerate(process_dl):
				X, names, paths, _ = batch
				with trc.set_grad_enabled(False):
					res = net.forward(X).data
				imgNames += names
				imgPaths += paths
				if res.dim() == 0: preds = trc.cat([preds, res.reshape(1)])
				else: preds = trc.cat([preds, res])
				
				app.processEvents()
				if self.abort_flag:
					self.aborted.emit()
					self.thread.quit()
					return
				self.progress.emit(5 + int(80/k / n_steps * (i+1)))

			if os.path.isdir(out): shutil.rmtree(out)
			os.mkdir(out)
			f, res = open(out_txt, 'w'), []
			for i, name in enumerate(imgNames):
				res.append((name, preds[i].item(), imgPaths[i]))
			res = sorted(res, key=lambda x: -x[1])
			for i in res:
				f.write(i[0]+' '+str(i[1])+'\n')
			f.close()

			app.processEvents()
			if self.abort_flag:
				self.aborted.emit()
				self.thread.quit()
				return
			self.progress.emit(90//k + 3)

			os.mkdir(os.path.join(out, 'No')), os.mkdir(os.path.join(out, 'Yes'))
			used_0, used_1 = {}, {}
			for i in res:
				name, ext = os.path.splitext(i[0])
				wdt = '.jgw' if ext == '.jpg' else '.tfw'
				world = os.path.splitext(i[2])[0] + wdt
				if i[1] > ui.sb_thld_cls.value()/100:
					if i[0] in used_1:
						name += '_' + str(used_1[i[0]])
						used_1[i[0]] += 1
					else:
						used_1[i[0]] = 1
					cls_d = os.path.join(out, 'Yes')
					sorted_img_paths.append(os.path.join(cls_d, name+ext))
				else:
					if i[0] in used_0:
						name += '_' + str(used_0[i[0]])
						used_0[i[0]] += 1
					else:
						used_0[i[0]] = 1
					cls_d = os.path.join(out, 'No')
				shutil.copy(i[2], os.path.join(cls_d, name+ext))
				shutil.copy(world, os.path.join(cls_d, name+wdt))
			
			app.processEvents()
			if self.abort_flag:
				self.aborted.emit()
				self.thread.quit()
				return
			self.progress.emit(90//k + 5)
			max_prog_before_seg = 50
		
		seg_sorted = opt_ui.cb_segSorted.isChecked() and ui.cb_CLS.isChecked()
		if ui.cb_SEG.isChecked() and ((seg_sorted and len(sorted_img_paths) > 0) or not seg_sorted):
			out = os.path.join(self.path_base, 'SEG')
			out_imgs, out_masks = os.path.join(out, 'Images'), os.path.join(out, 'Masks')

			if seg_sorted:
				dt_process = processDataset(sorted_img_paths, read_img_trfs)
			else:
				dt_process = processDataset(self.imgPaths, read_img_trfs)
			batch_size = min(128, len(dt_process))
			process_dl = trc.utils.data.DataLoader(dt_process, batch_size)

			n_steps = len(dt_process) // batch_size
			if len(dt_process) % batch_size != 0:
				n_steps += 1

			net = trc.load(NET_SEG)
			preds, imgNames, imgPaths, imgShapes = trc.tensor([]), [], [], []
			for i, batch in enumerate(process_dl):
				X, names, paths, shapes = batch
				with trc.set_grad_enabled(False):
					res = net.forward(X).data
				imgNames += names
				imgPaths += paths
				imgShapes += shapes
				if res.dim() == 0: preds = trc.cat([preds, res.reshape(1)])
				else: preds = trc.cat([preds, res])

				app.processEvents()
				if self.abort_flag:
					self.aborted.emit()
					self.thread.quit()
					return
				self.progress.emit(max_prog_before_seg + int(80/k / n_steps * (i+1)))

			if os.path.isdir(out): shutil.rmtree(out)
			os.mkdir(out), os.mkdir(out_imgs), os.mkdir(out_masks)
			used, prev_prg = {}, 0
			for i, name in enumerate(imgNames):
				shape = (imgShapes[0][i].item(), imgShapes[1][i].item())
				resize_trf = transforms.Resize(shape, antialias=False)
				mask = preds[i].clone()
				if not opt_ui.cb_rawMasks.isChecked():
					mask = (mask >= ui.sb_thld_seg.value()/100).int()*255.0
				mask = resize_trf(mask)
				
				base, ext = os.path.splitext(name)
				wdt = '.jgw' if ext == '.jpg' else '.tfw'
				world = os.path.splitext(imgPaths[i])[0] + wdt
				if name in used:
					base += '_' + str(used[name])
					used[name] += 1
				else:
					used[name] = 1
				img = base + ext
				sv_path_msk, sv_path_img = os.path.join(out_masks, img), os.path.join(out_imgs, img)
				save_image(mask, sv_path_msk)
				shutil.copy(imgPaths[i], sv_path_img)
				shutil.copy(world, os.path.splitext(sv_path_img)[0]+wdt)
				shutil.copy(world, os.path.splitext(sv_path_msk)[0]+wdt)

				app.processEvents()
				if self.abort_flag:
					self.aborted.emit()
					self.thread.quit()
					return
				prg = 90 + int(10 / len(imgNames) * (i+1))
				if prg != prev_prg:
					self.progress.emit(prg)
					prev_prg = prg

		print('Done!')
		self.progress.emit(100)
		self.finished.emit(str(datetime.timedelta(seconds=int(time.time()-start_time))))
		self.thread.quit()
	
	def abort(self):
		self.abort_flag = True

def open_SRC():
	'''filePath, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Open Image', os.path.expanduser('~/Desktop'),
		'PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*)')'''
	if ui.tabWidget.currentIndex() == 0:
		src_line, opt_key = ui.line_strSrc, 'src_last_1'
	else:
		src_line, opt_key = ui.line_strSrc_2, 'src_last_2'
	
	def_path = os.path.expanduser('~/Desktop')
	if opt_ui.comB_srcDef.currentIndex() == 0:
		with open(SETTINGS, 'r') as f:
			data = json.load(f)
			if opt_key in data and os.path.isdir(data[opt_key]):
				def_path = data[opt_key]
	elif os.path.isdir(src_line.text()):
		def_path = src_line.text()

	filePath = QtWidgets.QFileDialog.getExistingDirectory(window, 'Выберите корневую директорию поиска', def_path)
	if filePath:
		src_line.setText(filePath)
		with open(SETTINGS, 'r') as f:
			data = json.load(f)
			data[opt_key] = filePath
		os.remove(SETTINGS)
		with open(SETTINGS, 'w') as f:
			json.dump(data, f)

def scan_SRC():
	if ui.tabWidget.currentIndex() == 0:
		path, lst, grv, cb = ui.line_strSrc.text(), ui.list_foundImgs, ui.graphicsView, ui.cb_scanSubs
		fileTypes = ('.tif', '.tiff')
	else:
		path, lst, grv, cb = ui.line_strSrc_2.text(), ui.list_foundImgs_2, ui.graphicsView_2, ui.cb_scanSubs_2
		fileTypes = ('.tif', '.tiff', '.jpg')
	
	if os.path.isdir(path):
		files = []
		if cb.isChecked():
			for content in os.walk(path):
				for file in content[2]:
					if file[-4:] in fileTypes or file[-5:] in fileTypes:
						files.append(os.path.join(content[0], file))
		else:
			for file in os.listdir(path):
				if file[-4:] in fileTypes or file[-5:] in fileTypes:
					files.append(os.path.join(path, file))
		
		if len(files) == 0:
			QtWidgets.QMessageBox.information(window, 'Результаты поиска', 'Файлы не найдены...   ')
		else:
			grv.setScene(QtWidgets.QGraphicsScene())
			lst.model().clear()

			for file in files:
				item = QtGui.QStandardItem(file)
				item.setEditable(False)
				item.setCheckable(True)
				item.setCheckState(2)
				lst.model().appendRow(item)
			
			if len(files) == 1:
				QtWidgets.QMessageBox.information(window, 'Результаты поиска', 'В результате поиска был найден 1 файл.')
			elif str(len(files))[-1] in ('2', '3', '4') and str(len(files))[-2:] not in ('12', '13', '14'):
				QtWidgets.QMessageBox.information(window, 'Результаты поиска', 'В результате поиска было найдено '+str(len(files))+' файла.')
			else:
				QtWidgets.QMessageBox.information(window, 'Результаты поиска', 'В результате поиска было найдено '+str(len(files))+' файлов.')
	else:
		QtWidgets.QMessageBox.warning(window, 'Ошибка поиска', 'Такой директории не существует...')

def show_IMG():
	###############
	# add scaling #
	###############
	if ui.tabWidget.currentIndex() == 0:
		lst, grv = ui.list_foundImgs, ui.graphicsView
	else:
		lst, grv = ui.list_foundImgs_2, ui.graphicsView_2

	if len(lst.selectedIndexes()) > 0:
		image = QtGui.QImage(lst.selectedIndexes()[0].data())

		vp_w = grv.width()-30
		if image.width() > vp_w:
			pixmap = QtGui.QPixmap.fromImage(image).scaled(vp_w, vp_w)
		else:
			pixmap = QtGui.QPixmap.fromImage(image)
		
		grv.setScene(QtWidgets.QGraphicsScene())
		grv.scene().addPixmap(pixmap)

def set_OUT():
	if ui.tabWidget.currentIndex() == 0:
		res_line, opt_key = ui.line_strWrite, 'res_last_1'
	else:
		res_line, opt_key = ui.line_strWrite_2, 'res_last_2'
	
	def_path = os.path.expanduser('~/Desktop')
	if opt_ui.comB_srcDef.currentIndex() == 0:
		with open(SETTINGS, 'r') as f:
			data = json.load(f)
			if opt_key in data and os.path.isdir(data[opt_key]):
				def_path = data[opt_key]
	elif os.path.isdir(res_line.text()):
		def_path = res_line.text()
	
	filePath = QtWidgets.QFileDialog.getExistingDirectory(window, 'Выберите корневую директорию для записи результатов', def_path)
	if filePath:
		res_line.setText(filePath)
		with open(SETTINGS, 'r') as f:
			data = json.load(f)
			data[opt_key] = filePath
		os.remove(SETTINGS)
		with open(SETTINGS, 'w') as f:
			json.dump(data, f)

def cut_imgs():
	path_base = ui.line_strWrite.text()
	if os.path.isdir(path_base):
		model, imgPaths = ui.list_foundImgs.model(), []
		for i in range(model.rowCount()):
			item = model.item(i)
			if item.checkState() == 2: imgPaths.append(item.text())
		
		if len(imgPaths) > 0:
			ui.worker_cut = Worker_CUT(imgPaths, path_base, THD_CUT)
			ui.worker_cut.moveToThread(THD_CUT)
			THD_CUT.started.connect(ui.worker_cut.run)
			# ui.worker_cut.finished.connect(THD_CUT.quit)
			ui.popUp_cut = PopUpProgressBar(ui.worker_cut)
			ui.popUp_cut.show()
			THD_CUT.start()
		else:
			QtWidgets.QMessageBox.warning(window, 'Ошибка записи', 'Снимки не выбраны...')
	else:
		QtWidgets.QMessageBox.warning(window, 'Ошибка записи', 'Такой директории не существует...')

def net_prc():
	path_base = ui.line_strWrite_2.text()
	if os.path.isdir(path_base):
		if ui.cb_CLS.isChecked() or ui.cb_SEG.isChecked():
			model, imgPaths = ui.list_foundImgs_2.model(), []
			for i in range(model.rowCount()):
				item = model.item(i)
				if item.checkState() == 2: imgPaths.append(item.text())
			
			if len(imgPaths) > 0:
				ui.worker_nns = Worker_NNS(imgPaths, path_base, THD_NNS)
				ui.worker_nns.moveToThread(THD_NNS)
				THD_NNS.started.connect(ui.worker_nns.run)
				# ui.worker_nns.finished.connect(THD_NNS.quit)
				ui.popUp_nns = PopUpProgressBar(ui.worker_nns)
				ui.popUp_nns.show()
				THD_NNS.start()
			else:
				QtWidgets.QMessageBox.warning(window, 'Ошибка записи', 'Необходимо выбрать файлы для обработки...')
		else:
			QtWidgets.QMessageBox.information(window, 'Действия программы не заданы', 'Выберите необходимые функции обработки...')
	else:
		QtWidgets.QMessageBox.warning(window, 'Ошибка записи', 'Такой директории не существует...')

app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(window)
ui.tabWidget.setCurrentIndex(0)
ui.tabWidget.setTabEnabled(2, False)

if not os.path.isfile(SETTINGS):
	with open(SETTINGS, 'w') as f:
		json.dump({'_init_': 'do not edit this field'}, f)

# TAB_1 setup:
ui.list_foundImgs.setModel(QtGui.QStandardItemModel())
ui.list_foundImgs.selectionModel().selectionChanged.connect(show_IMG)

ui.graphicsView.setScene(QtWidgets.QGraphicsScene())
ui.graphicsView.setInteractive(True)
ui.graphicsView.setDragMode(1)

ui.btn_findSrc.clicked.connect(open_SRC)
ui.btn_startScan.clicked.connect(scan_SRC)
ui.btn_findWloc.clicked.connect(set_OUT)
ui.btn_start.clicked.connect(cut_imgs)

# TAB_2 setup:
ui.list_foundImgs_2.setModel(QtGui.QStandardItemModel())
ui.list_foundImgs_2.selectionModel().selectionChanged.connect(show_IMG)

ui.graphicsView_2.setScene(QtWidgets.QGraphicsScene())
ui.graphicsView_2.setInteractive(True)
ui.graphicsView_2.setDragMode(1)

ui.btn_findSrc_2.clicked.connect(open_SRC)
ui.btn_startScan_2.clicked.connect(scan_SRC)
ui.btn_findWloc_2.clicked.connect(set_OUT)
ui.btn_start_2.clicked.connect(net_prc)

# OPT setup:
opt_dialog = QtWidgets.QDialog()
opt_ui = Ui_DialogOptions()
opt_ui.setupUi(opt_dialog)
opt_ui.btn_clsOpt.clicked.connect(opt_dialog.hide)

# OTHER:
ui.act_opt.triggered.connect(opt_dialog.show)
ui.act_exit.triggered.connect(app.exit)

window.show()
sys.exit(app.exec_())
