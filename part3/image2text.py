#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: JUSHAH DCHODABA PYACHAM
# (based on skeleton code by D. Crandall, Oct 2020)
#

import os
import sys
import numpy as np
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

# Calculate initial probability and transition probability
##----------------------------------------------------------------------------
def calculate_trans_prob_init_prob(train_txt_fname):
    ## Initialize Train Letters
    TRAIN_LETTERS =  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    f = open(train_txt_fname, 'r');
    full_corpus = []

    ## Declaring counter for initial and transtion probability calculation
    init_freq = Counter({o:1 for o in TRAIN_LETTERS})
    trans_freq = {o1: Counter({o:1 for o in TRAIN_LETTERS})for o1 in TRAIN_LETTERS}
    
    ## Read each line from bc.train and attach to full corpus
    for line in f:
        # We have used bc.train so we are considering only the even positioned words of the line, for different text training file we can use the below commented line.
        data = [l for l in list(" ".join([w for w in line.split()][0::2]) ) if l in TRAIN_LETTERS ]
        full_corpus.append(data)

    ## First letter for each sentence. Calculate Initial Probability
    init_letters = [c[0] for c in full_corpus if len(c) > 0]
    init_freq.update(init_letters)
    init_prob = {i: init_freq[i]/sum(init_freq.values()) for i in init_freq.keys()}

    ## Creating tuple for i and i+1 items
    ## Also updating counter and calculating transition probabilities.
    trn = [(x,y) for corp in full_corpus for x,y in zip(corp, corp[1:]) ]
    for i,j in trn:
        trans_freq[i].update(j)        
    
    trans_prob = {i:{j: trans_freq[i][j]/sum(trans_freq[i].values()) for j in trans_freq[i].keys()}
                  for i in trans_freq.keys()}

    return init_prob, trans_prob
##----------------------------------------------------------------------------

# Emission Probability for HMM and Simplified net. It outputs log probabilties
def get_emission_p_ve(test_imge, train_char):
    
    ## Get train image from list of train letters
    train_img = train_letters[train_char]
    
    ## Count number of * and space in test image letter
    img_blk = sum([i.count('*') for i in test_imge[3:-1]])
    img_wht = len(test_imge[3:-1])*len(test_imge[0]) - img_blk
    
    ## take ratio of * to space
    factor = 999 if  img_blk == 0 else   img_wht / img_blk
    
    ## Iterate over each pixel in train and test and comapare them
    ## Naive Bayes assumption that our pixels are conditionally independent 
    ## given the letter.
    
    c = np.sum([np.sum([np.log(factor) if x=='*' and y=='*' else 0 for x,y in zip(test_imge[r], train_img[r])])
             for r in range(3, len(test_imge) - 1)])
    d = np.sum([np.sum([np.log(1 / np.sqrt(factor)) if x==' ' and y=='*' else 0 for x,y in zip(test_imge[r], train_img[r])])
             for r in range(3, len(test_imge) - 1)])

    return c+d
##----------------------------------------------------------------------------

### Simplified Net problem

def simplified(test_images):
    TRAIN_LETTERS =  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    seq = []
    for img in test_images:
        p_img_matches_chars = [(char, get_emission_p_ve(img, char)) for char
                               in TRAIN_LETTERS]  
        char_max_p = max(p_img_matches_chars, key=lambda x: x[1])[0]
        seq.append(char_max_p)
    return ''.join([char for char in seq])

##----------------------------------------------------------------------------

###  Hidden Markov Model ####
######## Below code is adapted from https://www.pythonpool.com/viterbi-algorithm-python/########
def hmm_viterbi(test_images):
    TRAIN_LETTERS =  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
   
    ## Empty Dictionary V for Viterbi Nodes
    
    V = [{}]
    
    ## Iterate over each hidden states (Train Letters)
    for char in TRAIN_LETTERS:
        
        ## Calculate Probability of states for first observation
        p_emit = get_emission_p_ve(test_images[0], char)
        p_init = p_initial[char]

        V[0][char] = {"prob": np.log(p_init) + p_emit, "prev": None}

    ## From 2nd node to last observation
    for t in range(1, len(test_images)):
        V.append({})
        
        ## Iterate over each hidden states (Train Letters)
        for char in TRAIN_LETTERS:
            
            ## Viterbi logic, iterate over each states and keep the one 
            ## with maximum for next observation
            max_tr_prob = V[t - 1][TRAIN_LETTERS[0]]["prob"] + np.log(p_transition[TRAIN_LETTERS[0]][char])
            prev_st_selected = TRAIN_LETTERS[0]
            for prev_type in TRAIN_LETTERS[1:]:
                tr_prob = V[t - 1][prev_type]["prob"] + np.log(p_transition[prev_type][char])
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_type

            ##Nultiply maximum of prev to emission probability of current
            max_prob = max_tr_prob + get_emission_p_ve(test_images[t], char)
            V[t][char] = {"prob": max_prob, "prev": prev_st_selected}
            
            
    ## Backtracking Viterbi Algorithm        
    opt = []
    max_prob = -1e9
    best_st = None

    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st


    for t in range(len(V) - 2, -1, -1):

        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    ## Final Result
    return "".join(opt)

  ##### Adaption of code ends here ################### 
p_initial, p_transition = calculate_trans_prob_init_prob(train_txt_fname)
simple_ans = simplified(test_letters)
hmm_ans = hmm_viterbi(test_letters)

 
# The final two lines of your output should look something like this:
print("Simple: " + simple_ans)
print("   HMM: " + hmm_ans) 


