# -*- coding: utf-8 -*-
"""
ESE 545 - Project 1
"""

import numpy as np
import pandas as pd
import scipy as sc
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import itertools
from itertools import combinations
import csv



## Function for creating shingles of size k
def nozip_ngrams(text, n):
    return set([text[i: i+n] for i in range(len(text)- n + 1)])

## Function to remove punctuation
def remove_punctuation(review_data):
  return ''.join([i for i in review_data if i not in punctuation])

## Function to remove stop words
def remove_stopwords(review_data):
  return ' '.join([item for item in review_data.split() if item not in stop_words])

## Function to compute Jaccard Distance between two arrays
def jaccard_distance(docu1, docu2):
  nonzero = np.bitwise_or(docu1!=0, docu2!=0)
  unequal_nonzero = np.bitwise_and((docu1!=docu2),nonzero)
  nn2 = np.double(unequal_nonzero.sum())
  nn4 = np.double(nonzero.sum())
  return (nn2/nn4) if nn4!=0 else 1

## Function to compute Jaccard Distance between two sparse matrices
def pairwise_sparse_jaccard_distance(X, Y):
    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)
    intersect = X.dot(Y.T)
    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = ((xx + yy).T - intersect)
    return (1 - intersect / union).A[0][0] if union != 0 else 1

## Function to find a random pair
def find_random_pair_doc():
  rand1 = np.random.randint(0,binary_matrix.shape[0]-1)
  rand2 = np.random.randint(0,binary_matrix.shape[0]-1)
  if rand1 != rand2:
    doca = binary_matrix.getrow(rand1)
    docb = binary_matrix.getrow(rand2)
  else:
    find_random_pair_doc()
  return doca, docb

## Function for permutation for min-hashing
def permute(aa,bb,rr, XX):
  mat=((aa*rr) + bb) % XX
  return mat.min(axis=1)

## Function for permutation for LSH
def row_permute(P, r):
  a_ = np.random.randint(1,P,r)
  b_ = np.random.randint(1,P,r)
  return a_.reshape(r,1), b_.reshape(r,1)


# Funciton for incoming review data
def find_nearest_review(input_review):
  binary_array = []
  shingles_list = []
  nearest_possible_reviews = []
  dist = 1
  nearest_doc = 0
  input_lsh_list = []
  nearest_doc = 0
  input_review = input_review.lower()
  input_review = remove_punctuation(input_review)
  input_review = remove_stopwords(input_review)
  input_shingles = nozip_ngrams(input_review, k)
  
  g_shingles_list = list(global_shingles_list)
  for shingle in g_shingles_list:
    if shingle in input_shingles:
      shingles_list.append(g_shingles_list.index(shingle))
  
  input_signature_matrix=(np.dot(a,np.array(shingles_list, ndmin=2)) + b) % R

  for i in g_shingles_list:
    if i in input_shingles:
      binary_array.append(1)
    else:
      binary_array.append(0)
      
  i_r = 5
  for band in range(0,20):
    band_array = ((a_list[band]*input_signature_matrix[band*i_r:(band+1)*i_r]) + b_list[band]) % P
    band_array = band_array.sum(axis=0)
    input_lsh_list.extend(band_array.tolist())
    
  for lsh_val in input_lsh_list:
    try:
      temp = band_dict[lsh_val]
      nearest_possible_reviews.extend(temp)
    except:
      pass

  for doc_i in nearest_possible_reviews:
    doc_i_binary = binary_matrix.getrow(doc_i).toarray()
    doc_i_jd = jaccard_distance(doc_i_binary, binary_array)
    if(dist > doc_i_jd):
      dist = doc_i_jd
      nearest_doc = doc_i
    
    return amazon_data['reviewerID'][nearest_doc]





if __name__ == "__main__":
     
    
    k = 5 # Number of characters in a shingle
    B = 20 # Number of bands in LSH
    r = 5 # Number of hashes in a band in LSH
    P = 987543637 # An enormous prime number used in LSH
    R = 476633  # of shingles = 476633, which is the greatest prime number to the number of shingles.
    m = B*r # Hash functions
    
    amazon_data = pd.read_json(r'amazonReviews.json', lines=True) # Read data
    
    col_list = ['reviewerID', 'reviewText'] # Drop every column except these two
    
    amazon_data=amazon_data[amazon_data['reviewText'].map(len) >= k] # Drop rows with review size less than k
    
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now","!", "\"", "#", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/",":",";", "<","=",">","?","@","[","\\","]","^","_","`","{","|","}","~"]
    
    
    amazon_data['reviewText'] = amazon_data['reviewText'].str.lower() # Convert reviews to lower case
    amazon_data['reviewText'] = amazon_data['reviewText'].apply(lambda x:remove_punctuation(x)) # Remove punctuation
    amazon_data['reviewText'] = amazon_data['reviewText'].apply(lambda x:remove_stopwords(x)) # Remove stop words
    
    
    
    amazon_data['review_shingles'] = amazon_data['reviewText'].apply(lambda x: nozip_ngrams(x, k))
    
    review_shingles_list = amazon_data['review_shingles']
    
    global_shingles_list = set.union(*review_shingles_list) # Creation of global shingle list
    
    ## Creation of Binary Sparse Matrix in which rows are documents and shingles are columns
    i = 0
    shingle_map = {}
    for shingle in global_shingles_list:
      shingle_map[shingle] = i
      i += 1
    rows = []
    cols = []
    r_list = []
    for i in range(0,amazon_data.shape[0]):
      try:
        r_list = [i]*len(review_shingles_list[i])
        rows.extend(r_list)
        for shingle in review_shingles_list[i]:
          try:
            index = shingle_map[shingle]
            cols.append(index)
          except:
            pass
      except:
        pass
    data = np.ones(len(rows))
    binary_matrix = csr_matrix((data, (rows, cols)), shape=(amazon_data.shape[0], len(shingle_map)), dtype=np.uint8)
    

    ## Compute Jaccard Distance list 
    jaccard_distance_list = []
    for i in range(0,10000):
      doc1, doc2 = find_random_pair_doc()
      jaccard_distance_list.append(pairwise_sparse_jaccard_distance(doc1, doc2))
    

    min_jaccard_dist = min(jaccard_distance_list)
    avg_jaccard_dist = sum(jaccard_distance_list)/len(jaccard_distance_list)
    print("Minimum Jaccard Distance = " + str(min_jaccard_dist))
    print("Average Jaccard Distance = " + str(avg_jaccard_dist))
    
    ## Histogram plot
    plt.figure()
    plt.hist(jaccard_distance_list)
    plt.xlabel('Jaccard Distance')
    plt.ylabel('Number of Pairs')
    
    # Justification for choice of r, b and m
    M = [1, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000] # The values of M taken into consideration
    Sim = np.arange(0,1,0.001) # The values of similarity (S)
    for i in M:
        r_val = [2, 5, 10, 20, 50, 100]
        b_val = [i/r for r in r_val]
        plt.figure()
        for r in range(len(r_val)):        
            plt.plot(Sim,1-(1-Sim**r_val[r])**b_val[r])
        plt.legend(labels = r_val, loc = 'upper left', title = "Value of r")
        plt.vlines(0.80, 0, 1, colors = "r", linestyles = "dotted")
        plt.xlabel('Similarity')
        plt.ylabel('Probability of Hit')
        plt.title("Number of Hash functions = %i" % i)
        plt.show()
        
    
    ## Min hashing
    index_dict = {key: [] for key in binary_matrix.nonzero()[0]}
    for row, col in zip(*binary_matrix.nonzero()):
      index_dict[row].append(col)
    
    a = []
    b = []
    for i in range(0,m):
      a.append(np.random.randint(0,R-1))
      b.append(np.random.randint(0,R-1))
    a = np.array(a, dtype=int).reshape([m,1])
    b = np.array(b, dtype=int).reshape([m,1])
    
    ## Compute signature matrix
    signature_matrix = np.zeros([m,binary_matrix.shape[0]],dtype=int)
    for doc_i in range(0,binary_matrix.shape[0]):
      index_list = index_dict.get(doc_i)
      if index_list != None:
        signature_matrix[:,doc_i] = permute(a,b,np.array(index_list, ndmin=2),R)    
    
    
    ## LSH
    row = 0
    band_list = []
    band_dict = {}
    a_list = []
    b_list = []
    for band in range(0,B):
      a1, b1 = row_permute(P, r)
      band_array = ((a1*signature_matrix[row:row+r,:]) + b1) % P
      band_array = band_array.sum(axis=0)
      band_array_list = band_array.tolist()
      ind = 0
      for b1 in band_array_list:
        try:
          temp = band_dict[b1]
          temp.append(ind)
          band_dict[b1] = temp
        except:
          band_dict[b1] = [ind]
        ind += 1
      band_list.extend(band_array_list)
      a_list.append(a1)
      b_list.append(b1)
    
    #to get unique hash values and its number of occurrences 
    band_array = np.array(band_list)
    hash_index,count = np.unique(band_array,return_index=False, return_inverse=False, return_counts=True)
    
    hash_index_list=hash_index.tolist()
    count_list=count.tolist()
    dictd = {k:v for k, v in zip(hash_index_list, count_list)}
    more_than_1 = []
    for key,val in dictd.items():
      if val > 1:
        more_than_1.append(key)
    

    doc_pair_list = []
    for hash_val in more_than_1:
      hash_bucket = band_dict[hash_val]
      if len(hash_bucket) > 2:
        comb = itertools.combinations(hash_bucket,2)
        for i in comb:
          doc_pair_list.append(i)
    
    doc_pair_array = np.array(doc_pair_list)
    
    
    ## Find all nearest neighbors
    nearest_neghbours = []
    for pair in doc_pair_array:
      d11 = binary_matrix.getrow(pair[0])
      d12 = binary_matrix.getrow(pair[1])
      jd1 = pairwise_sparse_jaccard_distance(d11,d12)
      if jd1 <= 0.2:
        nearest_neghbours.append(pair)
    
    ## Write nearest pairs to output.csv
    with open('output.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(nearest_neghbours)
    
    ## Test with new data
    nearest_neighbor = find_nearest_review('INPUTREVIEWNAME')
    print("Nearest neighbor is: ", nearest_neighbor)
