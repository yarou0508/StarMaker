# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:04:26 2018

@author: 47532
"""

import re
import regex
import tensorflow as tf
import pandas as pd
import numpy as np
import copy
from tensorflow.contrib import rnn
from tqdm import tqdm
import time
import sys

#%% ===================================处理数据 ============================= 
def get_data(num):
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
                sentences.append(word) 
            else:
                continue  
    words_sort = pd.DataFrame(words)[0].value_counts() 
    del words
    words_sort = words_sort[words_sort>1]                        
    word_bank = list(words_sort.index)
    del words_sort
    #word_bank = list(set(words))
    word2id = {}  # word => id 的映射
    for i in range(len(word_bank)):
        word2id[word_bank[i]] = i+1 
    word2id['!'] = len(word_bank)+1 # Word2id中增加‘EOS'
    word2id['$'] = len(word_bank)+2
    del word_bank
    inputs = []  
    for sent in sentences: # 输入是多个句子，这里每个循环处理一个句子
        input_sent = [len(word2id)]
        for i in range(sent.__len__()):  # 处理单个句子中的每个单词
            input_id = word2id.get(sent[i])
            if not input_id:  # 如果单词不在词典中，则跳过
                continue
            input_sent.append(input_id)
        input_sent.append(len(word2id)-1) # 每个句子末尾添加'EOS'
        if len(input_sent) > 22:
            input_sent = input_sent[:22]
            #input_sent = [input_sent[i:i+20] for i in range(0,len(input_sent),20)]
            #inputs.extend(input_sent)
            inputs.append(input_sent)
        elif len(input_sent) < 3:
            continue
        else:
            inputs.append(input_sent)
    del sentences
    pad = len(max(inputs, key=len))
    inputs = [i + [0]*(pad-len(i)) for i in inputs]
    return pad, inputs, len(word2id)+1, word2id


class TrainData:
    def __init__(self, inputs, batch_size, sen_length):
        self.inputs = inputs
        self.batch_size  = batch_size
        self.sen_length = sen_length
        self.n = len(inputs)       
    def get_batch_data(self, batch):
        global batch_size
        start_pos = batch * self.batch_size
        end_pos = min((batch + 1) * self.batch_size, self.n)
        xdata = self.inputs[start_pos:end_pos]
        # target data 左移一位
        ydata = copy.deepcopy(self.inputs[start_pos:end_pos])
        for row in ydata:
            row.pop(0)
            row.append(0)      
        x_batch = np.array(xdata, dtype=np.int32)
        y_batch = np.array(ydata, dtype=np.int32)
        return x_batch, y_batch
    def get_num_batches(self):
        return max(self.n, 0) // self.batch_size

def build_output_g(lstm_output, hidden_dim, softmax_w, softmax_b):
    outputs = tf.reshape(tf.concat(lstm_output,1), [-1, hidden_dim])      
    # 将lstm层与softmax层全连接,计算logits
    logits = tf.matmul(outputs, softmax_w) + softmax_b    
    # softmax层返回概率分布
    preds = tf.nn.softmax(logits, name='predictions')   
    return preds

def build_output_d(lstm_output, hidden_dim, sigmoid_w, sigmoid_b):
    outputs = tf.reshape(tf.concat(lstm_output,1), [-1, hidden_dim])      
    # 将lstm层与softmax层全连接,计算logits
    logits = tf.matmul(outputs, sigmoid_w) + sigmoid_b    
    return logits

#%% ===================================GAN 模型 ============================= 
class GAN:    
    def __init__(self, vocab_size, batch_size,
                 num_batches, num_steps, hidden_dim=64, 
                 num_layers=4, learning_rate=0.001, lambd=0.01, dropout_rate=0.2,  train=False):
        #self.graph = tf.Graph()
        self.lr = learning_rate
        self.lambd = lambd
        self.dropout_rate = dropout_rate        
        self.num_layers = num_layers  
        self.num_batches = num_batches
        self.hidden_dim = hidden_dim # BasicLSTM model needs input size = [batch_size, hidden_dim], therefore, embedding_size must = hidden_dim
        self.vocab_size = vocab_size
        if train == False:
            self.batch_size, self.num_steps = 1, 1
        else:
            self.batch_size, self.num_steps = batch_size, num_steps
        
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')    
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True            
        tf.reset_default_graph()  
        
# 输入层
        self.inputs = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='targets')
        # 创建softmax weight 和 bias
        with tf.variable_scope('softmax'):
            self.softmax_w1 = tf.get_variable("softmax_w1", shape = [self.hidden_dim, self.vocab_size], 
                                             regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd / self.num_batches), 
                                              initializer = tf.random_uniform_initializer(-1,1,seed=1))
            # tf.truncated_normal_initializer(mean = 0, stddev=1.0 / np.sqrt(self.hidden_dim),seed=1)
            self.softmax_b1 = tf.get_variable("softmax_b1", initializer = tf.zeros([self.vocab_size]))
            self.sigmoid_w1 = tf.get_variable("sigmoid_w1", shape = [self.hidden_dim, 1], 
                                             regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd / self.num_batches), 
                                              initializer = tf.random_uniform_initializer(-1,1,seed=1))
            # tf.truncated_normal_initializer(mean = 0, stddev=1.0 / np.sqrt(self.hidden_dim),seed=1)
            self.sigmoid_b1 = tf.get_variable("sigmoid_b1", initializer = tf.zeros([1]))
        W = tf.get_variable("word_embedding", shape=[self.vocab_size, self.hidden_dim],trainable=False)
        self.embedding = tf.placeholder(tf.float32, shape=[self.vocab_size, self.hidden_dim], name='embedding')
        self.embedding_init = tf.assign(W, self.embedding)
        # 对输入进行one-hot编码         
        self.inputs_emb = tf.nn.embedding_lookup(W, self.inputs)
        self.inputs_emb_unstack = tf.unstack(self.inputs_emb, self.num_steps, 1)  

        with tf.variable_scope("LSTM_g"):
            # Train contents
            self.lstm_cell_train = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, forget_bias=1.0, state_is_tuple=True)  
            # 添加dropout
            if train == True:
                self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_train, output_keep_prob=(1 - self.dropout_rate))
            # 堆叠
            self.lstm_cell_train= tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell_train] * self.num_layers, state_is_tuple=True)
            self.initial_state_train = self.lstm_cell_train.zero_state(self.batch_size, tf.float32) 
            
            # 运行RNN
            self.outputs_train, self.final_state_train = rnn.static_rnn(self.lstm_cell_train, self.inputs_emb_unstack, 
                         initial_state = self.initial_state_train, dtype=tf.float32)                
            # softmax prediction probability
            self.prediction_train = build_output_g(self.outputs_train, self.hidden_dim, self.softmax_w1, self.softmax_b1)
            self.out_index = tf.argmax(self.prediction_train,1)
            self.fake = tf.reshape(self.out_index,[-1, self.num_steps])
            self.fake_emb = tf.nn.embedding_lookup(W, self.fake)
        
        with tf.variable_scope("RNN_d"):
#            # 堆叠LSTM层方法一
#            self.lstm_cell_dis = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, forget_bias=1.0, state_is_tuple=True)  
#        # 添加dropout
#            if train == True:
#                self.lstm_cell_dis = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_dis, output_keep_prob=(1 - self.dropout_rate))
#            self.lstm_cell_dis= tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell_dis] * self.num_layers, state_is_tuple=True)
#            self.initial_state_dis = self.lstm_cell_dis.zero_state(self.batch_size, tf.float32) 
#            
#            def discriminator(sentence, lstm_cell, initial_state):
#                outputs, final_state = rnn.static_rnn(lstm_cell, sentence, 
#                             initial_state, dtype=tf.float32)
#                logits = build_output_d(outputs[:,-1,:], self.hidden_dim, self.sigmoid_w1, self.sigmoid_b1)
#                
#                return logits, final_state
            # 堆叠LSTM层方法二
            stack_rnn = []
            for i in range(3):
                stack_rnn.append(tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True))
            self.lstm_cell_dis = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple = True)
            self.initial_state_dis = self.lstm_cell_dis.zero_state(self.batch_size, dtype=tf.float32)
        
            def discriminator(lstm_cell, sentence,  initial_state_dis):
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, sentence, initial_state=initial_state_dis, time_major=False)
                predictions = build_output_d(outputs[:,-1,:], self.hidden_dim, self.sigmoid_w1, self.sigmoid_b1)
                return predictions, final_state
            self.logits_real, self.final_state_dis1 = discriminator(self.lstm_cell_dis, self.inputs_emb, self.initial_state_dis)
            self.logits_fake, self.final_state_dis = discriminator(self.lstm_cell_dis, self.fake_emb,  self.final_state_dis1)

        
        with tf.name_scope('train_generator'):
            self.g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.logits_fake), self.logits_fake)
            self.optimizer_g = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.g_loss)
        
        with tf.name_scope('train_discriminator'):
            self.d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.logits_real), self.logits_real)
            self.d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.logits_fake), self.logits_fake)
            self.d_loss = self.d_loss_real + self.d_loss_fake     
            self.optimizer_d = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.d_loss)
 
#%% ===================================训练数据 =============================            
if __name__ == "__main__":
    num = str(sys.argv[1])
    k = sys.argv[2]
    sen_length, inputs, vocab_size, word2id = get_data(num)
    batch_size = 32
    train_data = TrainData(inputs, batch_size, sen_length) #inputs, batch_size, sen_length
    print('Train size: %s' % (train_data.get_num_batches()*batch_size))
    print('Vocab_size: %s' % vocab_size)            
    num_batches = train_data.get_num_batches()
    model = GAN(vocab_size, batch_size, num_batches, sen_length, train=True)  
    word_embedding = np.load('./comments_model/skip_gram/%s/word_embedding.npy' % num)
    word_embedding = word_embedding.reshape([vocab_size,64])
    epochs = 50
    starttime = time.time()
    with tf.Session(config = model.config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        writer = tf.summary.FileWriter('./comments_model/GAN/%s' % num, sess.graph)  # self.global_step.eval(session=sess)
        step = 0
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)  
        embedding = sess.run([model.embedding_init],feed_dict={model.embedding: word_embedding})
        for index in range(epochs):      
            total_loss_g = 0.0
            total_loss_d = 0.0
            new_state_train, new_state_dis = sess.run([model.initial_state_train, model.initial_state_dis])
            for batch in tqdm(range(num_batches)):
                batch_inputs,batch_targets = train_data.get_batch_data(batch)
                if batch % (k+1) != 0: 
                    feed = {model.inputs: batch_inputs, model.targets: batch_targets, model.initial_state_train: new_state_train, model.initial_state_dis: new_state_dis}
                    batch_loss_g, new_state_train, _ = sess.run([model.g_loss, model.final_state_train, model.optimizer_g], feed_dict=feed)
                    total_loss_g += batch_loss_g
                else:
                    feed = {model.inputs: batch_inputs, model.targets: batch_targets, model.initial_state_train: new_state_train, model.initial_state_dis: new_state_dis}
                    batch_loss_d, new_state_dis, _ = sess.run([model.d_loss, model.final_state_dis, model.optimizer_d], feed_dict=feed)
                    total_loss_d += batch_loss_d
                step += 1
                if step % 100 == 0:
                    saver.save(sess, './comments_model/GAN/%s/' % num, global_step=step)
                    #if step % 10000 == 0:
                        #print(step)
            print('G and D Loss at step {}: {:5.6f}, {:5.6f}'.format(index+1, total_loss_d / num_batches, total_loss_g / num_batches))
    endtime = time.time()  
    print(endtime - starttime) 

#%% ===================================生成句子 ============================= 
def pick_top_n(preds, vocab_size, iter, top_n=10):
    # 从预测结果中选取前top_n个最可能的字符
    p = np.squeeze(preds)
    #p1 = list(p)
    #c = p1.index(max(p1))
    if iter <= 3:
        p = p[1:-2]
    # 将除了top_n个预测值的位置都置为0
        p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
        p = p / np.sum(p)
    # 随机选取一个字符
        c = np.random.choice(vocab_size-3, 1, p=p)[0]+1
    else:
        p = p[1:-1]
    # 将除了top_n个预测值的位置都置为0
        p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
        p = p / np.sum(p)
    # 随机选取一个字符
        c = np.random.choice(vocab_size-2, 1, p=p)[0]+1
    return c
        
def sample(n_words, vocab_size, batch_size, num_batches, sen_length, prime, num):
    #prime: 起始文本    
    samples=[]
    # sampling=True意味着batch的size=1 x 1
    model = GAN(vocab_size, batch_size, num_batches, sen_length, train=False)
    saver = tf.train.Saver()
    with tf.Session(config = model.config) as sess:
        # 加载模型参数，恢复训练
        checkpoint_file = tf.train.latest_checkpoint('./comments_model/GAN/%s' % num)
        saver.restore(sess, checkpoint_file)
        new_state_train = sess.run([model.initial_state_train])        
        # 不断生成字符，直到达到指定数目
        c = word2id.get(prime)
        for i in range(n_words):
            test_word_id = c
            if test_word_id == word2id.get('!'):
                break
            else:
                feed = {model.inputs: [[test_word_id]],
                        model.initial_state_train: new_state_train}
                preds, new_state_train = sess.run([model.prediction_train, model.final_state_train], feed_dict=feed)
                c = pick_top_n(preds, vocab_size, i)
                while c == test_word_id:
                    c = pick_top_n(preds, vocab_size, i)   
                samples.extend(x for x,v in word2id.items() if v==c)      
    print(' '.join(samples))
    
if __name__ == "__main__":
    num = str(sys.argv[1])
    sen_length, inputs, vocab_size, word2id = get_data(num)
    batch_size = 30
    train_data = TrainData(inputs, batch_size, sen_length) #inputs, batch_size, sen_length
    print('Train size: %s' % (train_data.get_num_batches()*batch_size))
    # Train size: 795392
    print('Vocab_size: %s' % vocab_size)  
    # Vocab_size: 81685            
    num_batches = train_data.get_num_batches()    
    for i in range(10):  
        #sample(10, vocab_size, batch_size, num_batches, prime = "you")
        sample(20, vocab_size, batch_size, num_batches, sen_length, '$', num)
