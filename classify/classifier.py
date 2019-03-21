import fasttext

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
class Classifier(object):

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025,probability=True),
        SVC(gamma=2, C=1,probability=True),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    ]
        #MLPClassifier(alpha=1)]
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM","Random Forest"]


    def train_fasttext(self,train, path):
        """
        Train fastext classifier
        :param train:training data
        :param path: path for saving the model
        :return: classifier
        """

        classifier = fasttext.supervised(train, path, label_prefix='__label__', epoch=200, lr=0.5, word_ngrams=1)
        return classifier

    def load_fastext_model(self, path):
        """
        load saves fastext classifier

        :param path:
        :return: classifier
        """
        classifier = fasttext.load_model(path)
        return classifier

    def train_tfidf(self,train,path):
        """
        perform tfidf vectorization
        :param train: data
        :param path: path to save
        :return: tfidf vectors
        """
        vectorizer = TfidfVectorizer(stop_words='english')
        model = vectorizer.fit_transform(train).toarray()
        joblib.dump(model, path)

        return model


    def all_classifiers(self,X_train,y_train,test):
        """
        Perfrom classification on several classifiers
        :param X_train: training data
        :param y_train: training labels
        :param test: test data
        :return:
        """
        predictions=[]
        for name, clf in zip(self.names, self.classifiers):
            #print(name)
            clf.fit(X_train, y_train)
            p = clf.predict_proba(test)
            predictions.append((clf.classes_,p))
        return predictions

