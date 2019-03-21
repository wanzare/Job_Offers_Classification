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


## To Run

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

