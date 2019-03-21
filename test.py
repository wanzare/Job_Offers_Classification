
from test_wanzare import main
import itertools
from test_wanzare.utils import inOut as io

def test_model(args):
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
    main.create_train_test(data,path)

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

    model = main.fasttext.supervised(train, model_path, epoch=200)
    labels = model.predict_proba(texts)

    # compare different classifiers
    if compare:
        clf_labels = main.compare_classifiers(trains, train_labels,texts, path)
        clf_labels["Fasttext"] =[" ".join(x[0][0].split("_")) for x in labels]


    # estimate quality
    #result = model.test(train)
    #print ('P@1:', result.precision)
    #print ('R@1:', result.recall)

    avg_quality=[]
    for x,y in itertools.combinations(clf_labels.keys(),2):

        sim = sum(main.compare_listcomp(clf_labels.get(x), clf_labels.get(y))) / len(texts)
        print(x, " <=> ", y," : ", sim )
        avg_quality.append(sim)
        #
    print("Estimated quality : ", sum(avg_quality)/len(list(itertools.combinations(clf_labels.keys(),2))))

    # save results for the fasttext classifier to json file
    main.save_results(path,labels,test_data,use_title,thresh)

