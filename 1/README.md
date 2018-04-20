# ULL Practical 1: Evaluating Word Representations

To view the results, open the .ipynb file. 

### Reading the data

The code is this sections loads the different models into memroy

### Word similarity

The code in this section expects the word embeddings to be in memory. If that
is the case, the correlations can be obtained by running the cells in sequence.

### Analogies

The code in this sections also expects the word embeddings to be in memory. Again,
obtaining the results is straight forward. Just run the cells under the section
"Word Analogy". The second computes the MRR and accuracy (the sample size of the experiment has
been set to 100 for a managable execution time) and the third prints the
predictions for a hand picked subset of the analogies. 

### Clustering

The number of clusters can be specified, and is used to draw the plots. We implemented 
both the PCA and TSNE algorithm, and plots for both can be shown. Due to computational contraints,
we only draw the TSNE plot for the dependency based model. The second section prints
word clusters with the same words for all three models, to compare results. 