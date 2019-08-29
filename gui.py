from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import os
import subprocess
import threading

def runThread(target,args=()):
	my_thread = threading.Thread(target=target,args=args)
	my_thread.start()

def newButton(parent,text="",icon=None,slot=None):
	but = QPushButton(text,parent)
	if icon:
		but.setIcon(icon)
	if slot:
		but.clicked.connect(slot)

	return but

class Gui(QMainWindow):
	def __init__(self,parent=None):
		super(Gui,self).__init__(parent)

		widget = QWidget()
		self.setCentralWidget(widget)


		layout = QVBoxLayout()

		controls = ["Capture","Extract","Train SVC","Test","Demo Camera","Plot embeddings"]
		slots = [self.capture,self.extract,self.trainSVC,self.test,self.demo,self.plot]

		for name,slot in zip(controls,slots):
			but = newButton(self,name,slot=slot)
			layout.addWidget(but)

		self.centralWidget().setLayout(layout)

	def capture(self):
		cmd = "python camera.py"
		print("please wait a few seconds ... ")
		runThread(target=lambda:os.system(cmd))

	def extract(self):
		cmd = "python extract.py"
		print("please wait a few seconds ... ")
		runThread(target=lambda:os.system(cmd))
	
	def trainSVC(self):
		d = input("Enter max embeddings : ")
		try:
			d = int(d)
		except:
			d = 20

		emb_path = input("embeddings path : ")
		model_path = input("model path : ")

		cmd = "python svc_model.py -m train -e %s -model %s -n %d"%(emb_path,model_path,d)
		print("please wait a few seconds ... ")
		runThread(target=lambda:os.system(cmd))

	def test(self):
		cmd = "python svc_model.py -m test -model model"
		print("please wait a few seconds ... ")
		runThread(target=lambda:os.system(cmd))

	def demo(self):
		cmd = "python camera_demo.py 0.4"
		print("please wait a few seconds ... ")
		runThread(target=lambda:os.system(cmd))

	def plot(self):
		cmd = "python plot_embeddings.py"
		print("please wait a few seconds ... ")
		runThread(target=lambda:os.system(cmd))
		
		# MyOut  = subprocess.Popen(cmd, shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
		# stdout,stderr = MyOut.communicate()
		# print(stdout)


if __name__ == "__main__":
	import sys
	app = QApplication(sys.argv)
	window = Gui()
	window.show()
	app.exec_()