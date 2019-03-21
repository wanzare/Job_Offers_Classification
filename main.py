import sys

#sys.path.append('/path/to/project/folder')
from test_wanzare.classify import classifier as cl
from test_wanzare.utils import inOut as io
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from nltk.corpus import stopwords
import itertools
import ast
import fasttext
import numpy as np

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def tokenize(texts):
    """
    Tokenize texts
    :param texts: text
    :return: tokens
    """
    word_tokens = tokenizer.tokenize(texts.strip())  # tokenize the text
    vocab = [w.lower() for w in word_tokens if w not in stop_words]

    return vocab


def create_train_test(data,path):
    """
    Use the examples with level as training data and those without as test data
    :param data: json file containing the data
    :param path: path for saving files
    :return:
    """
    io.create_directory(path)
    train_text = open(path+"/train.txt", "w") # saves texts to be used for training the classifier
    test_text = open(path+"/test.txt", "w") # saves texts with missing level
    test_data = open(path+"/raw_test.txt", "w") # saves the raw data with missing levels
    train_data = open(path + "/raw_train.txt", "w")  # saves the raw data with levels
    train_labels =  open(path + "/train_labels.txt", "w")  # saves the train labels

    train_titles=defaultdict(list)
    test_titles = defaultdict()
    count =0
    for doc in data:
        texts = doc["description"]
        title = doc["title"]
        title = tokenize(title)
        try:
            level = doc["level"]
        except KeyError:
            level = "None"
            pass
        text = " ".join(tokenize(texts.encode("ascii", "replace").decode("ascii")))

        if text == "no description available" or text == "": # ignore texts with no descriptions provided
            continue
        # print(level)
        if level == "None":
            test_text.write(text + "\n")
            test_data.write(str(doc) + "\n")
            test_titles[count]=title
            count+=1
        else:
            train_titles[level].append(title)
            train_data.write(text + "\n")
            train_labels.write(level+"\n")
            train_text.write("__label__" + "_".join(level.split(" ")) + " " + text + "\n")

    train_title_path = path + "/train_titles.txt"  # save titles from texts with levels
    test_title_path = path + "/test_titles.txt" # save the titles for the texts without levels
    io.save_file(train_titles, train_title_path)
    io.save_file(test_titles, test_title_path)
    train_text.close()
    test_text.close()
    test_data.close()





def build_title_features(titles):
    """
    Finds the relationshiop of words found in the title to seniority levels in order to augement the results
    from the classifier.

    I apply bayes rule to get relationship of seniority (S) given word (w):
    P(S|w) = P(w|S)*P(S) / P(w)
    :param titles: job titles
    :return: dictionary with
    """

    titles_dict= defaultdict(int) # holds the counts of each word in the titles
    #
    for t in titles:
        key_words = [x for x in itertools.chain(*titles.get(t))]
        for word in key_words:
            titles_dict[word] += 1

    total_words = sum(titles_dict.values()) # sum of total words in the titles

    titles_dict_norm = defaultdict() # holds the normalized counts of each word in thr title

    for t in titles:
        #print(t)
        key_dict = defaultdict(int) # holds the counts of each word in the given seniority category
        key_dict_norm = defaultdict(float) # # holds the normalized counts of each word in the given seniority category
        key_words= [x for x in itertools.chain(*titles.get(t))]
        for word in key_words:
            key_dict[word]+=1
        total_t_words = sum(key_dict.values()) # # sum of total words in the category

        # use bayes inference to get the probability of the category given word
        for word in key_words:
            #print(key_dict.get(word), titles_dict.get(word))
            # TODO : normalization
            weight = ((key_dict.get(word)/total_t_words*1.0)*0.25)/(titles_dict.get(word)/total_words)

            key_dict_norm[word]= weight

        titles_dict_norm[t]=key_dict_norm



    return titles_dict_norm



def use_title_features(title,titles_dict_norm):
    """
    Predict the seniority using the title features
    :param title: title words
    :param titles_dict_norm: dictionary with weights
    :return: level prediction
    """
    title_pred=[]

    for k in titles_dict_norm:
        count = defaultdict(float)
        for w in title:
            if w in titles_dict_norm.get(k):
                #print(k,w,titles_dict_norm.get(k).get(w))
                count[k] += titles_dict_norm.get(k).get(w)

        if count.get(k) is not None:
            title_pred.append((count.get(k),k))


    title_pred.sort(reverse=True)



    return title_pred

def compare_listcomp(x, y):
    """
    compare two lists element by element and return similarity count
    :param x: list
    :param y: list
    :return: list
    """
    return [1 for i, j in zip(x, y) if i == j]


def compare_classifiers(trains,texts,path):
    """
    Classify the test data againg several classifiers
    :param trains:training data
    :param texts: test data
    :param path: path tosave
    :return: predictions for each classifier
    """
    classif = cl.Classifier()
    tf_data = classif.train_tfidf(trains+texts, path + "/tf.data")
    tf_train= tf_data[:len(trains)]
    tf_test = tf_data[len(trains):]
    names = classif.names
    predictions = classif.all_classifiers( tf_train, train_labels, tf_test)
    #print(predictions)
    clf_labels = {}
    for x in range(len(predictions)):
        clf = predictions[x]
        labels=[]
        #print(clf[0])
        #print(clf[1])

        for pred in clf[1]:
            i = np.argmax(pred)
            labels.append(clf[0][i])
        clf_labels[names[x]]= labels
    return clf_labels

def save_results(labels,use_title,thresh):
    """
    Save predictions to json file
    :param labels: predicted labels
    :param use_title: boolean to use title features or not
    :param thresh: threshold for using the title features
    :return:
    """
    save_test=[]

    for item, label in zip(test_data, labels):

        job = ast.literal_eval(item)
        title = tokenize(job["title"]) # load titles
        # use title features
        if use_title:

            titles = io.load_file(path + "/titles.txt")
            titles_dict_norm = build_title_features(titles)
            # predict label using title feature
            if label[0][1] < thresh:

                t_label = use_title_features(title,titles_dict_norm)
                if len(t_label) > 0:
                    job["level"] = t_label[0][1]
                    #print("HERE", t_label[0], label, title)
                else:
                    job["level"] = " ".join(label[0][0].split("_"))


            else:
                job["level"] = " ".join(label[0][0].split("_"))
                #print(label,title)
            #print("---")

        else:
            job["level"] = " ".join(label[0][0].split("_"))
        save_test.append(job)
    #print(len(save_test))

    io.save_file(save_test,path+"/label_pred.json",indent=4)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, default="data",
                        help="path to directory for saving data")
    parser.add_argument("-data", type=str,
                        help="path to directory containin the data")

    parser.add_argument("-model", type=str, default="model",
                        help="path to directory for saving the classifier model")
    parser.add_argument('-t',  default=True,

                        help='Set use of title features to False')
    parser.add_argument('-c', default=True,

                        help='Compare different classifer outputs')
    parser.add_argument('-thresh', type=float,default=0.8,
                        help='Threshold for using title features') # TODO: Tune threshold

    args = parser.parse_args()
    print(args)
    compare = args.c
    thresh = args.thresh
    model_path = args.model
    if args.data is None:
        raise ValueError("Provide path to the data file")
    else:
        data = io.load_file(args.data)
    path = args.path
    use_title = args.t

    # split into training and testing
    create_train_test(data,path)

    # load train data
    train_labels = open(path+"/train_labels.txt","r").readlines()
    train_labels = [d.strip() for d in train_labels]
    train = path+"/train.txt"
    train_data = open(path + "/raw_train.txt", "r")
    trains = [d.strip() for d in train_data.readlines()]

    # load test data
    test_data = open(path+"/raw_test.txt","r")
    test_data = test_data.readlines()
    t = open(path + "/test.txt", "r")
    texts = [d.strip() for d in t.readlines()]


    # fastext model

    model = fasttext.supervised(train, model_path, epoch=200)
    labels = model.predict_proba(texts)

    # compare different classifiers
    if compare:
        clf_labels = compare_classifiers(trains, texts, path)
        clf_labels["Fasttext"] =[" ".join(x[0][0].split("_")) for x in labels]


    # estimate quality
    #result = model.test(train)
    #print ('P@1:', result.precision)
    #print ('R@1:', result.recall)

    avg_quality=[]
    for x,y in itertools.combinations(clf_labels.keys(),2):

        sim = sum(compare_listcomp(clf_labels.get(x), clf_labels.get(y))) / len(texts)
        print(x, " <=> ", y," : ", sim )
        avg_quality.append(sim)
        #
    print("Estimated quality : ", sum(avg_quality)/len(list(itertools.combinations(clf_labels.keys(),2))))

    # save results for the fasttext classifier to json file
    save_results(labels, use_title,thresh)

