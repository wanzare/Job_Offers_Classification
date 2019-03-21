### Augment offer information
Given the list of job offers (title, description and seniority level),
but some of the offers are missing the seniority level.

Write an application which fills in the gaps - restores the missing seniority level.

Extra:

1. Explain the choice of language / technology stack.
2. Explain the choice of approach and algorithm.
3. Estimate quality of the result.


## Choice of language / technology stack
I use python3.6 because there are several off the shelf tools for performing text processing and macine learning available in python.

I approach the task as a classification probelm: I use the job offers with seniority levels are training data.

For text preprocessing, I use NLTK toolkit. NLKT has several functions for text processing and analysis.

I use several classifiers for classificatiion, in order to compare the performance across the classifiers and estimate quality of predictions.

As input to the classifiers, I represent the preprocessed as documents as tfidf vector and word embeddings from Facebooks fasttext pretrained word vectors. Both representations have been shown to be useful for classification.


## Choice of approach and algorithm.
I decided to approach the task as a classification task because of availability of some label data that could be used for classification. As a classification task, we are able to quickly try out different classifiers in order to get compe up with possible directions for improving the system.

## Estimate quality of the result.
I compare predictions between all pairs of classifiers and estimate the quality of the results by taking the average. My hypotheses is that If the different classifiers have comparable agreement, then the results are more reliable than if the classifiers do not agree i.e. agreement would be close to chance / random level.

I would perform error analysis and annotation on a sample of texts where the different classifiers tend not to agree and also where they tend to agree.




The expected outtput does a pairwise comparison between classifiers as output below:
Nearest Neighbors  <=>  Linear SVM  :  0.4794520547945205
Nearest Neighbors  <=>  RBF SVM  :  0.4931506849315068
Nearest Neighbors  <=>  Random Forest  :  0.3835616438356164
Nearest Neighbors  <=>  Fasttext  :  0.5753424657534246
Linear SVM  <=>  RBF SVM  :  0.9863013698630136
Linear SVM  <=>  Random Forest  :  0.9041095890410958
Linear SVM  <=>  Fasttext  :  0.5342465753424658
RBF SVM  <=>  Random Forest  :  0.8904109589041096
RBF SVM  <=>  Fasttext  :  0.547945205479452
Random Forest  <=>  Fasttext  :  0.4383561643835616


Similar classifier e.g. Linear SVM  <=>  RBF SVM  have higher prediction agreement as compares to classifiers with different architectures e.g. Nearest Neighbors  <=>  Random Forest  :  0.3835616438356164.

From the results, we can see that Nearest Neighbors has the least agreement with other classifiers and is therefore not a good model for this task. On average RBF SVM has the highest agreement with other models, followed by Linear SVM. I would argue that an SVM based model would be a better model for this task.

The final estimate for quality is printed out:

Estimated quality :  0.6232876712328768

The final estimate for quality is 0.623. Although this is just raw agreement,  it still indicates that the approach is significantly higher than random assignment as the models agree on average 62% of the time. 

## To Run

 Requirements:

- install nltk
- install numpy
- install scipy
- install sklearn
- import nltk
- nltk.download("stopwords")

 Add the path to the project to PYTHONPATH:

import sys
sys.path.append('/path/to/project')

To run the code
```bash
python main.py -data path/to/data.json
```
with the below options:

    "-path", default="data",
                        help="path to directory for saving data")
    "-data", type=str,
                        help="path to directory containin the data")

    "-model", type=str, default="model",
                        help="path to directory for saving the classifier model")
    '-t',  default=True,
    			 help='Set use of title features to False')

    '-c', default=True,
                        help='Compare different classifer outputs')

    '-thresh', type=float,default=0.8,
                        help='Threshold for using title features')

The model outputs several files in the path provided using -path:
 - label_pred.json : stores the results of the prediction

