###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids: jushah, pyacham, dchodaba
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
from collections import Counter
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        
        ## Declaring Part of speech set
        self.pos_type = ['noun', 'adj', 'adv', 'adp', 'conj', 'det', 'num',
                         'pron', 'prt', 'verb', 'x', '.']
        
        ## Delcaring different probabilities
        ## We will update their values in train function
        
        ## Initial probability
        self.initial_probability = Counter({o: 1.0 for o in self.pos_type}) 
        self.total_pos = Counter({o: 1.0 for o in self.pos_type})
        self.prob_si = {}
        self.total_word = 0
        self.frequency_dict = {o: {} for o in self.pos_type}
        
        # P(word | S) -- Emission probability
        self.prob_wi_si = {o: {} for o in self.pos_type} 
        
        # P(s1 | s0) -- Transition probability
        self.freq_si1_si = {o: {key: 0 for key in self.pos_type} for o in
                            self.pos_type}
        self.prob_si1_si = {o: {key: 0 for key in self.pos_type} for o in
                            self.pos_type} 
        
        # P(s2 | s1, s0)-- Transition probability for complex net
        self.freq_si2_si = {o: {o1 : {key: 1 for key in self.pos_type} for
                                o1 in self.pos_type} for o in self.pos_type}
        self.prob_si2_si = {o: {o1 : {key: 0 for key in self.pos_type} for
                                o1 in self.pos_type} for o in self.pos_type}

        # P(w1 | s1, s0)-- Emission probability for complex net
        self.freq_wi1_si = {o: {o1 : {} for o1 in self.pos_type} for o in
                            self.pos_type}
        self.prob_wi1_si = {o: {o1 : {} for o1 in self.pos_type} for o in
                            self.pos_type}

    
    
    
    
    
    # Calculate the log of the posterior probability of a given sentence

    def posterior(self, model, sentence, label):
        if model == "Simple":
            return self.simplified_prob(sentence, label)
        elif model == "HMM":
            return self.viterbi_prob(sentence, label)
        elif model == "Complex":
            return self.mcmc_probability(sentence, label)
        else:
            print("Unknown algo!")


    ## Training functions
    def train(self, data):
        
        ## Initial probability calcualtion using Collections Counter
        self.initial_probability.update([i[1][0] for i in data])
        self.initial_probability= {o: self.initial_probability[o] / sum(
            self.initial_probability.values()) for o in self.pos_type}
        
        ## Emission Probabilities
        self.total_pos.update([item for m in [list(i[1]) for i in data] for
                               item in m])
        self.total_word = len([item for m in [list(i[0]) for i in data] for
                               item in m])
        self.prob_si = {key: self.total_pos[key] / self.total_word for 
                        key in self.pos_type}

        ## Transition and emission probability for HMM and Complex Net
        for item in data:
            
            for i in range(len(item[0])):
                try:
                    self.frequency_dict[item[1][i]][item[0][i]] += 1
                except KeyError:
                    self.frequency_dict[item[1][i]][item[0][i]] = 1
            
                if i != len(item[0]) - 1:                   
                    self.freq_si1_si[item[1][i]][item[1][i + 1]] += 1
                if i < len(item[0]) - 2:
                    self.freq_si2_si[item[1][i]][item[1][i + 1]][item[1][i + 2]] += 1
                if i != len(item[0]) - 1:
                    try:
                        self.freq_wi1_si[item[1][i]][item[1][i + 1]][item[0][i+1]] += 1
                    except KeyError:
                        self.freq_wi1_si[item[1][i]][item[1][i + 1]][item[0][i+1]] = 1
                

        for key in self.pos_type:
            for elements, value in self.frequency_dict[key].items():
                self.prob_wi_si[key][elements] = value / sum(list(self.frequency_dict[key].values()))
            for k1, val1 in self.freq_si1_si[key].items():
                prob = val1 / sum(list(self.freq_si1_si[key].values()))
                self.prob_si1_si[key][k1] = prob if prob != 0 else 0.00000001
                
            for k1 in self.freq_si2_si[key].keys():                
                for k2, val1 in self.freq_si2_si[key][k1].items():
                    prob = val1 / sum(list(self.freq_si2_si[key][k1].values()))
                    self.prob_si2_si[key][k1][k2] = prob if prob != 0 else 0.00000001

            for k1 in self.freq_wi1_si[key].keys():                
                for elements, value in self.freq_wi1_si[key][k1].items():
                    prob = value / sum(list(self.freq_wi1_si[key][k1].values()))
                    self.prob_wi1_si[key][k1][elements] = prob if prob != 0 else 0.00000001  
                    
                    
                    
    def simplified(self, sentence):
        ## Returns max of probabilitties for each POS tag.
        ## Returns sequence of POS tag for each word in sentence based on probabilities
        ## Simplified net assumes neighboring POS tag are independent of each other
        for word in sentence:
            seq =  [max([(word, type, np.log(self.prob_wi_si[type][word]) +
                          np.log(self.prob_si[type])) if word in
                         self.prob_wi_si[type] else (word, type,np.log(1e-9))
                         for type in self.pos_type  ], key=lambda x:x[2])
                    for word in sentence]
        
        
        return [s[1] for s in seq], sum([s[2] for s in seq])

    def simplified_prob(self, sentence, label):
       return sum([max([np.log(self.prob_wi_si[type][word]) +  np.log(self.prob_si[type]) if word in self.prob_wi_si[type]
                               else np.log(1e-9) for type in label]) for word in sentence])

######## Below code is adapted from
######## https://www.pythonpool.com/viterbi-algorithm-python/########
### Viterbi Algorithm for HMM ##########
    def hmm_viterbi(self, sentence):
        
        ## Declare empty tree for Viterbi
        V = [{}]
        
        for type in self.pos_type:
            try:
                p_emit = self.prob_wi_si[type][sentence[0]]
            except:
                ## If not assume very small probability  number to avoid 
                ## zero log error. This is due to unknown word
                p_emit = 0.00000001
            
            ## First Node , starting node of Viterbi for each hidden states
            V[0][type] = {"prob": np.log(self.initial_probability[type])
                          + np.log(p_emit), "prev": None}
        
        ## From 2nd to end node, traverse through each hidden states
        for t in range(1, len(sentence)):
            V.append({})
            for type in self.pos_type:
                max_tr_prob = V[t - 1][self.pos_type[0]]["prob"] + np.log(
                    self.prob_si1_si[self.pos_type[0]][type])
                prev_st_selected = self.pos_type[0]
                for prev_type in self.pos_type[1:]:
                    tr_prob = V[t - 1][prev_type]["prob"] + np.log(
                        self.prob_si1_si[prev_type][type])
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_type
                        
                
                try:
                    p_emit = self.prob_wi_si[type][sentence[t]]
                except:
                    p_emit = 0.00000001
                
                ## Multiply max of previous to current emission
                ## and store for each hidden states at every node
                max_prob = max_tr_prob + np.log(p_emit)
                V[t][type] = {"prob": max_prob, "prev": prev_st_selected}
        
        ## Backtracking for viterbi starts here    
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


        return opt, max_prob

  ##### Adaption of code ends here ################### 

    def viterbi_prob(self, sentence, label):
        try:
            p_emit = self.prob_wi_si[label[0]][sentence[0]]
        except:
            p_emit = 0.00000001
            
        p_viter = np.log(self.initial_probability[label[0]]) + np.log(p_emit)

        for i in range(1, len(label)):
            
                p_viter += np.log(self.prob_si1_si[label[i-1]][label[i]])
                
                try:
                    p_emit = self.prob_wi_si[label[i]][sentence[i]]
                except:
                    p_emit = 0.00000001
                
                p_viter += np.log(p_emit)
                
        return p_viter


    # returns the posterior for the complex modelvusing gibbs sampling 
    ## Reference :  https://people.duke.edu/~ccc14/sta-663/MCMC.html
    def mcmc_probability(self, sentence, sample):
        # P(S1)* {P(S2/S1)...P(Sn/Sn-1)*{P(W1/S1)....P(Wn/Sn)}{P(S3/S1, S2)
        #      ....P(Sn/Sn,Sn-1}{P(W2/S1, S2)....P(Wn/Sn-1, Sn)}

        s1 = sample[0]
        prob_s1 = np.log(self.initial_probability[s1])
        a, b ,c ,d  = (0,0,0,0)

        for i in range(len(sample)):
            # P(Wi | Si)
            try:
                p_emit = self.prob_wi_si[sample[i]][sentence[i]]
            except:
                p_emit = 0.00000001  
            
            # P(Wi | Si, Si-1)
            try:
                p_emit2 = self.prob_wi1_si[sample[i - 1]][sample[i]][sentence[i]]
            except:
                p_emit2 = 0.00000001
            
            c += np.log(p_emit)
            d += np.log(p_emit2)
            
            if i != 0:
                b += np.log(self.prob_si1_si[sample[i - 1]][sample[i]])
            if i > 1:
                a += np.log(self.prob_si2_si[sample[i - 2]][sample[i - 1]][sample[i]])

        return prob_s1+a+b+c+d  
   
    # Generating Gibbs samples
    def generate_sample(self, sentence, sample):
        tags = self.pos_type
        for index in range(len(sentence)):
            p_array = [0] * len(self.pos_type)
            log_p_array = [0] * len(self.pos_type)
            
            ## Generating Gibbs sample by iterating through Each POS Tag
            for j in range(len(self.pos_type)):
                sample[index] = tags[j]
                log_p_array[j] = self.mcmc_probability(sentence, sample)

            a = min(log_p_array)
            log_p_array = [j-a for j in log_p_array]
            p_array = [np.exp(j) for j in log_p_array]

            s = sum(p_array)
            p_array = [x / s for x in p_array]
            rand = random.random()
            p = 0
            for i in range(len(p_array)):
                p += p_array[i]
                if rand < p:
                    sample[index] = tags[i]
                    break
        return sample
    
    ## Main code for solving complex MCMC model
    ## Reference https://aclanthology.org/C10-2016.pdf
    def complex_mcmc(self, sentence):
        samples = []
        count_tags_array = []
        # Initial is based on simplified probabilities
        sample = self.simplified(sentence)[0] 
        
        ## Number of iteration, all sample after 50 iteration is considered
        n_iter= 12 
        b_iter = 3

        for i in range(n_iter):
            sample = self.generate_sample(sentence, sample)
            if i >= b_iter :  samples.append(sample)

        for j in range(len(sentence)):
            count_tags = {}
            for sample in samples:
                try:
                    count_tags[sample[j]] += 1
                except KeyError:
                    count_tags[sample[j]] = 1
            count_tags_array.append(count_tags)

        final_tags = [max(count_tags_array[i], key = count_tags_array[i].get)
                      for i in range(len(sentence))]
        return [ tag for tag in final_tags ]     

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.

    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)[0]
        elif model == "HMM":
            return self.hmm_viterbi(sentence)[0]
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

