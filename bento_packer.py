from sklearn import svm
from sklearn import datasets

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X,y = iris.data, iris.target 
clf.fit(X,y)

from bento_service import IrisClassifier1
iris_service_packer = IrisClassifier1() 
iris_service_packer.pack("model",clf)

saved_path = iris_service_packer.save()