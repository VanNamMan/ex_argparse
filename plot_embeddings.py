import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import pickle
import itertools
import hashlib

from PyQt5.QtGui import QColor

# from libs.utils import generateColorByText

# folder = "embdings/"

# vec = pickle.load(open("embdings/NguyenDucThanh.pickle","rb"))
# print(vec.shape)

def generateColorByText(text):
    s = str(text)
    hashCode = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hashCode / 255) % 255)
    g = int((hashCode / 65025)  % 255)
    b = int((hashCode / 16581375)  % 255)
    return QColor(r, g, b, 100)


def _plot_embds(folder_embds):

	vecs = []
	labels = []
	for f in os.listdir(folder_embds):
		filename = os.path.join(folder_embds,f)
		vec = pickle.load(open(filename,"rb"))
		print(vec.shape)
		vecs.append(vec)
		labels.append(f.split(".")[0])


	
	X = list(itertools.chain.from_iterable(vecs))
	print(len(X),labels)

	pca = PCA(n_components=3).fit(X)
	print(pca)

	fig = plt.figure(figsize=(10,8))
	ax = fig.add_subplot(111, projection='3d')
	plt.rcParams['legend.fontsize'] = 10  

	for vec,label in zip(vecs,labels):
		xd = pca.transform(vec)
		ax.plot(xd[:,0], xd[:,1], xd[:,2],
	        'o', markersize=8, color=generateColorByText(label+".pickle").name(), alpha=0.7, label=label)


	plt.title('Embedding Vector')
	ax.legend(loc='upper right')
	plt.savefig("Embedding_Vector.png")
	plt.show()

if __name__ == "__main__":
	folder = input("folder of embeddings : ")
	_plot_embds(folder)
