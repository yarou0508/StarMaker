# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:16:33 2018

@author: 47532
"""
import re
import regex
import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import sys

#%% ===================================Clean data============================= 
def get_data(win_len, num):
    file = "comment-%s"
    with open(file % num, encoding='utf8', mode='r') as rfile:
        words = []
        sentences = []
        def repl(m):
            inner_word = list(m.group(0))
            return " " + ''.join(inner_word) + " "
        for line in rfile:
            line = line.lower()
            if ('id=' in line) or ('|||' in line) or ('>>>' in line) or ('•' in line) or ('●' in line) or ('╭━╮' in line):
                continue    
            #line = re.sub(r'<.*>', ' ', line)
            line = re.sub('[\=\s+\.\!\?\;\,\/\\\_\。\$\%^*(+\"\:\-\@\#\&\|\[\]\<\>)]+', " ", line)
            line = re.sub('\d{10}', " ", line)
            sentence =  regex.sub(r'\p{So}\p{Sk}*', repl, line)
            word = sentence.split()                     
            if len(word) > 1:
                if "'" in word:
                    word.remove("'")
                elif "''" in word:
                    word.remove("''")
                else: 
                    word = word
                words.extend(word)
                word.insert(0,'$')
                word.extend('!')
                sentences.append(word) 
            else:
                continue  
    words_sort = pd.DataFrame(words)[0].value_counts() 
    del words
    # Adjust the word corpus size by limit the word occurence frequency
    words_sort = words_sort[words_sort>2]                        
    word_bank = list(words_sort.index)
    del words_sort
    word2id = {}  # word => id 
    for i in range(len(word_bank)):
        word2id[word_bank[i]] = i+1
    word2id['!'] = len(word_bank)+1# add 'EOS' to Word2id
    word2id['$'] = len(word_bank)+2
    inputs = []
    labels = []
    for sent in sentences:  # Loop to process sentence by sentence
        for i in range(sent.__len__()):  # Loop to process word by word in one sentence
            start = max(0, i - win_len)  # set the window as [-win_len,+win_len], total 2*win_len+1 in length
            end = min(sent.__len__(), i + win_len + 1)
            # Create input word and its corresponding target words
            for index in range(start, end):
                if index == i:
                    continue
                else:
                    input_id = word2id.get(sent[i])
                    label_id = word2id.get(sent[index])
                    if not (input_id and label_id):  
                        continue
                    inputs.append(input_id)
                    labels.append(label_id)
    del sentences
    return word_bank, word2id, len(word2id)+1, inputs, labels

class TrainData:
    def __init__(self, inputs, labels, word_bank, vocab_size):
        self.word_bank = word_bank
        self.vocab_size = vocab_size
        self.inputs = inputs
        self.labels = labels
        self.n = len(inputs)       
    def get_batch_data(self, batch):
        global batch_size
        start_pos = batch * batch_size
        end_pos = min((batch + 1) * batch_size, self.n)
        batch_inputs = self.inputs[start_pos:end_pos]
        batch_labels = self.labels[start_pos:end_pos]           
        batch_inputs = np.array(batch_inputs, dtype=np.int32)
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])
        return batch_inputs, batch_labels
    def get_num_batches(self):
        return max(self.n - 1, 0) // batch_size + 1

#%% ===================================skip-gram model ============================= 
class Skip_gram:
    def __init__(self, vocab_size, embedding_size, num_batches, lambd, num_sampled, learning_rate):
        self.graph = tf.Graph()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_batches = num_batches
        self.lambd = lambd
        self.batch_size = None
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.build_graph()
    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size],name = 'train_inputs')
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name = 'train_labels')
            self.test_word_id = tf.placeholder(tf.int32, shape=[None], name = 'test_word_id')
    def _create_embedding(self):
        with tf.name_scope("embed"):
            self.embedding_dict = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size],
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd / self.num_batches),
                                     initializer=tf.random_uniform_initializer(-1,1,seed=1))
            # Initialize weight matrix
            self.nce_weight = tf.get_variable('weight', shape = [self.vocab_size, self.embedding_size], regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd / self.num_batches), initializer = tf.truncated_normal_initializer(mean = 0, stddev=1.0 / np.sqrt(self.embedding_size),seed=1))
            #tf.truncated_normal_initializer(mean = 0, stddev=1.0 / np.sqrt(self.embedding_size),seed=1)
            # Initialize the bais
            self.nce_biases = tf.get_variable('biases', initializer = tf.zeros([self.vocab_size]))
            #tf.Variable(tf.zeros([self.vocab_size]), name = "biases")
    def _create_loss(self):
        with tf.name_scope("loss"):
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs)  # batch_size
            # Compute the soft-max loss
#            self.loss = tf.reduce_mean(
#                    tf.nn.sampled_softmax_loss(
#                    weights=self.nce_weight,  
#                    biases=self.nce_biases,  
#                    labels=self.train_labels, 
#                    inputs=embed,  # input embedding vector
#                    num_sampled=self.num_sampled,  # negative sampling number
#                    num_classes=self.vocab_size, # total number of words
#                    remove_accidental_hits=True))
            # Compute the NCE loss
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weight, 
                    biases=self.nce_biases,  
                    labels=self.train_labels, 
                    inputs=embed, # input embedding vector
                    num_sampled=self.num_sampled,  # negative sampling number
                    num_classes=self.vocab_size, # total number of words
                    remove_accidental_hits=True
                )
            )               
    def _create_evaluation(self):
        with tf.name_scope("evaluation"):           
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_dict), 1, keepdims=True))
            self.normed_embedding = self.embedding_dict / norm
            test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
            self.similarity = tf.matmul(test_embed, tf.transpose(self.normed_embedding), name = 'similarity')
    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    def build_graph(self):
        with self.graph.as_default() as g:
            with g.device('/cpu:0'):
                self._create_placeholders()
                self._create_embedding()
                self._create_loss()
                self._create_evaluation()
                self._create_optimizer()
                self._create_summaries()
    def train(self, train_data,num_train_steps=10, num='1'):
        os.system('rm -rf ./comments_model/skip_gram/%s' % num)
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            writer = tf.summary.FileWriter('./comments_model/skip_gram/%s' % num, sess.graph)
            initial_step = 0  # self.global_step.eval(session=sess)
            step = 0
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            for index in range(initial_step, initial_step + num_train_steps):
                total_loss = 0.0
                for batch in tqdm(range(self.num_batches)):
                    batch_inputs,batch_labels = train_data.get_batch_data(batch)
                    # 　feed in the training data
                    feed_dict = {self.train_inputs: batch_inputs,
                        self.train_labels: batch_labels}
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],feed_dict=feed_dict)
                    total_loss += loss_batch
                    step += 1
                    if step % 100 == 0:
                        writer.add_summary(summary, global_step=step)
                        saver.save(sess, './comments_model/skip_gram/%s/' % num, global_step=step)
                        if step % 10000 == 0:
                            print(step)
                print('Train Loss at step {}: {:5.6f}'.format(index, total_loss / train_data.get_num_batches()))
            word_embedding = sess.run([self.embedding_dict])
            np.save('./comments_model/skip_gram/%s/word_embedding.npy' % num, word_embedding)

#%% ===================================Train the model ============================= 
if __name__ == "__main__":
    num = str(sys.argv[1])
    word_bank, word2id, vocab_size, inputs, labels = get_data(3, num)
    batch_size = 1024                
    train_data = TrainData(inputs, labels, word_bank, vocab_size) # win_len, inputs, word_bank, vocab_size
    print('Train size: %s' % train_data.n)
    print('Vocab_size: %s' % vocab_size)                
    model = Skip_gram(vocab_size, 64, train_data.get_num_batches(), 0.01, 10, 0.001)    # vocab_size, embedding_size, num_batches, lambd, num_sampled, learning_rate           
    starttime = time.time()
    model.train(train_data, 30, num=num) # train_data, epochs, num
    endtime = time.time()  
    print(endtime - starttime)  

#%% ===================================Evaluate word embedding ============================= 
def predict(test_words, num):
    sess = tf.Session()
    checkpoint_file = tf.train.latest_checkpoint('./comments_model/skip_gram/%s' % num)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file), clear_devices=True)
    saver.restore(sess, checkpoint_file)
    graph = sess.graph
    g_test_word_id = graph.get_operation_by_name("data/test_word_id").outputs[0]
    g_similarity = graph.get_operation_by_name("evaluation/similarity").outputs[0]      
    test_word_id = [word2id.get(x) for x in test_words]
    feed_dict = {g_test_word_id: test_word_id}
    similarity = sess.run(g_similarity, feed_dict=feed_dict)
    top_k = 8
    for i in range(len(test_words)):
        nearest = (-similarity[i, :]).argsort()[1:top_k+1]        
        log = "Nearest to '%s':" % test_words[i]
        for k in range(top_k):
            close_word = [x for x,v in word2id.items() if v == nearest[k]]
            log = '%s %s,' % (log, close_word)
        print(log)

if __name__ == "__main__":
    num = str(sys.argv[1])
    word_bank, word2id, vocab_size, inputs, labels = get_data(3, num)
    batch_size = 1024                
    train_data = TrainData(inputs, labels, word_bank, vocab_size) # win_len, inputs, word_bank, vocab_size
    print('Train size: %s' % train_data.n)
    print('Vocab_size: %s' % vocab_size)                
    test_words = ['nice', 'perfect' ,'song', 'superb', '😘', 'voice', 'i','bro', 'sis', 'looks', 'sounds', 'both']
    
    predict(test_words, num)

