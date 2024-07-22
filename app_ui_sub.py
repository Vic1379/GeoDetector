from PyQt5 import QtCore, QtGui, QtWidgets

class PopUpProgressBar(QtWidgets.QDialog):
	abort = QtCore.pyqtSignal()
	def __init__(self, worker):
		super().__init__()
		h, w, center = 150, 550, QtWidgets.QDesktopWidget().availableGeometry().center()
		self.layout = QtWidgets.QVBoxLayout(self)
		self.setFixedSize(w, h)
		self.setModal(True)

		self.bar = QtWidgets.QProgressBar(self)
		self.bar.setAlignment(QtCore.Qt.AlignCenter)

		self.label = QtWidgets.QLabel('Пожалуйста дождитесь окончания выполнения программы...', self)
		self.label.setAlignment(QtCore.Qt.AlignTop)

		self.btn_ok, self.btn_cancel = QtWidgets.QPushButton('Продолжить', self), QtWidgets.QPushButton('Отмена', self)
		self.btn_ok.setEnabled(False)
		self.btn_ok.clicked.connect(self.hide)
		self.btn_cancel.clicked.connect(self.abort.emit)
		
		btnsWidget, btnsLayout = QtWidgets.QWidget(), QtWidgets.QHBoxLayout()
		btnsLayout.setContentsMargins(200, 10, 0, 0)
		btnsLayout.addWidget(self.btn_ok), btnsLayout.addWidget(self.btn_cancel)
		btnsWidget.setLayout(btnsLayout)
		
		self.layout.addWidget(self.bar)
		self.layout.addWidget(self.label)
		self.layout.addWidget(btnsWidget)

		self.setGeometry(center.x()-(w+1)//2, center.y()-h//2, w, h)
		self.setWindowTitle('Идёт обработка')
		# self.show()

		# connect signals with the worker
		worker.progress.connect(self.on_progress_changed)
		worker.aborted.connect(self.hide)
		worker.finished.connect(self.finish)
		self.abort.connect(worker.abort)
	
	def finish(self, time_str):
		self.btn_cancel.setEnabled(False)
		self.btn_ok.setEnabled(True)
		self.label.setText('Готово! Работа программы завершена за '+time_str+'.')

	def on_progress_changed(self, value):
		print(value)
		self.bar.setValue(value)
