### Augment offer information
Given the list of job offers (title, description and seniority level),
but some of the offers are missing the seniority level.

Write an application which fills in the gaps - restores the missing seniority level.

Extra:

1. Explain the choice of language / technology stack.
2. Explain the choice of approach and algorithm.
3. Estimate quality of the result.


## choice of language / technology stack
I use python3.6 because there are several off the shelf tools for performing text processing and macine learning available in python. 

I approach the task as a classification probelm: I use the job offers with seniority levels are training data.

For text preprocessing, I use NLTK toolkit. NLKT has several functions for text processing and analysis. 

I use several classifiers for classificatiion, in order to compare the performance across the classifiers and estimate quality of predictions.

As input to the classifiers, I represent the preprocessed as documents as tfidf vector and word embeddings from Facebooks fasttext pretrained word vectors. Both representations have been shown to be useful for classification. 


## choice of approach and algorithm.
I decided to approach the task as a classification task because of availability of some label data that could be used for classification. 

## Estimate quality of the result.
I compare predictions between all pairs of classifiers and estimate the quality of the results by taking the average. 