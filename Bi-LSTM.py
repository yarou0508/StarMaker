# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:01:46 2018

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

def get_data():
    with open('comment1', encoding='utf8', mode='r') as rfile:
        words = []
        sentences = []
        def repl(m):
            inner_word = list(m.group(0))
            return " " + ''.join(inner_word) + " "
        for line in rfile:
            line = line.lower()
            line = re.sub(r'<.*>', ' ', line)
            line = re.sub('[\s+\.\!\?\,\/_,$%^*(+\"\:\-\@\#\&)]+', " ", line)
            sentence =  regex.sub(r'\p{So}\p{Sk}*', repl, line)
            word = sentence.split()                        
            if len(word) > 1:
                if "'" in word:
                    word.remove("'")
                else: 
                    word = word
                words.extend(word)
                sentences.append(word) 
            else:
                continue  
    words_sort = pd.DataFrame(words)[0].value_counts() 
    #words_sort = words_sort[words_sort>1]                        
    word_bank = list(words_sort.index)
    #word_bank = list(set(words))
    word2id = {}  # word => id 的映射
    for i in range(len(word_bank)):
        word2id[word_bank[i]] = i+1 
    word2id['EOS'] = len(word_bank)+1 # Word2id中增加‘EOS'
    inputs = []   
    for sent in sentences: # 输入是多个句子，这里每个循环处理一个句子
        input_sent = []
        for i in range(sent.__len__()):  # 处理单个句子中的每个单词
            input_id = word2id.get(sent[i])
            if not input_id:  # 如果单词不在词典中，则跳过
                continue
            input_sent.append(input_id)
        input_sent.append(len(word_bank)+1) # 每个句子末尾添加'EOS'
        if len(input_sent) > 21:
            input_sent = [input_sent[i:i+20] for i in range(0,len(input_sent),20)]
            inputs.extend(input_sent)
        else:
            inputs.append(input_sent)
    #pad = np.mean([len(x) for x in inputs])
    pad = len(max(inputs, key=len))
    inputs = [i + [0]*(pad-len(i)) for i in inputs]
    return word_bank, pad, inputs, len(word2id)+1, word2id
word_bank, sen_length, inputs, vocab_size, word2id = get_data()

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
            b = row.pop(0)
            row.append(b)      
        x_batch = np.array(xdata, dtype=np.int32)
        y_batch = np.array(ydata, dtype=np.int32)
        return x_batch, y_batch
    def get_num_batches(self):
        return max(self.n - 1, 0) // self.batch_size 
    
batch_size = 128
train_data = TrainData(inputs, batch_size, sen_length) #inputs, batch_size, sen_length
print('Train size: %s' % (train_data.get_num_batches()*batch_size))
print('Vocab_size: %s' % vocab_size)  

def build_inputs(num_steps):
    # num_seqs: 每个batch中的序列个数
    # num_steps: 每个序列包含的字符数
    inputs = tf.placeholder(tf.int32, shape=[None, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, shape=[None, num_steps], name='targets')
    return inputs, targets

def build_lstm(hidden_dim, num_layers, batch_size, dropout_rate,sampling):
    # hidden_dim: lstm隐层中结点数目
    # num_layers: lstm的隐层数目
    # lstm cell
    lstm_cell_fw = rnn.BasicLSTMCell(hidden_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_bw = rnn.BasicLSTMCell(hidden_dim, forget_bias=1.0, state_is_tuple=True)   
    # 添加dropout
    if sampling == False:
        lstm_cell_fw = rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
        lstm_cell_bw = rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
    # 堆叠
    lstm_cell_fw = rnn.MultiRNNCell([lstm_cell_fw] * num_layers, state_is_tuple=True)
    lstm_cell_bw = rnn.MultiRNNCell([lstm_cell_bw] * num_layers, state_is_tuple=True)
    initial_state_fw = lstm_cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = lstm_cell_bw.zero_state(batch_size, tf.float32)    
    return lstm_cell_fw, lstm_cell_bw, initial_state_fw, initial_state_bw

def build_output(lstm_output, hidden_dim, vocab_size, lambd, num_batches):
    outputs = tf.reshape(tf.concat(lstm_output,1), [-1, hidden_dim * 2])
    # 将lstm层与softmax层全连接
    with tf.variable_scope('softmax'):
        softmax_w = tf.get_variable("softmax_w", shape = [hidden_dim * 2, vocab_size], 
                                              regularizer=tf.contrib.layers.l2_regularizer(scale=lambd / num_batches), 
                                              initializer = tf.random_uniform_initializer(-1,1,seed=1))
        softmax_b = tf.get_variable("softmax_b", initializer = tf.zeros([vocab_size])) 
    # 计算logits
    logits = tf.matmul(outputs, softmax_w) + softmax_b    
    # softmax层返回概率分布
    preds = tf.nn.softmax(logits, name='predictions')   
    return preds, logits

def build_loss(logits, targets):      
    # Softmax cross entropy loss   
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(targets, [-1]), logits = logits))   
    return loss

def build_optimizer(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)   
    return optimizer

class Bi_LSTM:    
    def __init__(self, vocab_size, batch_size, 
                 num_batches, num_steps=sen_length, embedding_size=32, hidden_dim=30, 
                 num_layers=1, learning_rate=0.01, lambd=0.01, dropout_rate=0.5,  sampling=True):
        self.lr = learning_rate
        self.lambd = lambd
        self.dropout_rate = dropout_rate        
        self.num_layers = num_layers  
        self.num_batches = num_batches
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim       
        self.vocab_size = vocab_size
        if sampling == True:
            self.batch_size, self.num_steps = 1, 1
        else:
            self.batch_size, self.num_steps = batch_size, num_steps
                    
        tf.reset_default_graph()        
        # 输入层
        self.inputs = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='targets')
        # LSTM层
        lstm_cell_fw, lstm_cell_bw, self.initial_state_fw, self.initial_state_bw = build_lstm(self.hidden_dim, self.num_layers, self.batch_size, self.dropout_rate, sampling)
        # 对输入进行one-hot编码
        self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size],
                                 regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd / self.num_batches),
                                 initializer=tf.random_uniform_initializer(-1,1,seed=1))
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.inputs_emb = tf.unstack(self.inputs_emb, self.num_steps, 1)
        # 运行RNN
        outputs, self.final_state_fw, self.final_state_bw = rnn.static_bidirectional_rnn(lstm_cell_fw, lstm_cell_bw, self.inputs_emb, 
                     initial_state_fw = self.initial_state_fw, initial_state_bw = self.initial_state_bw, dtype=tf.float32)
        # softmax prediction probability
        self.prediction, self.logits = build_output(outputs, self.hidden_dim, self.vocab_size, self.lambd, self.num_batches)
        # Loss 和 optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets)
        self.optimizer = build_optimizer(self.loss, self.lr)
        
#%% ===================================训练数据 ============================= 
num_batches = train_data.get_num_batches()
model = Bi_LSTM(vocab_size, batch_size, num_batches, sampling=False)

epochs = 20
with tf.Session() as sess:
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    writer = tf.summary.FileWriter('./comments_model/bi_LSTM', sess.graph)  # self.global_step.eval(session=sess)
    step = 0
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    new_state_fw, new_state_bw = sess.run([model.initial_state_fw, model.initial_state_bw])
    for index in range(epochs):      
        total_loss = 0.0
        for batch in tqdm(range(num_batches)):
            batch_inputs,batch_targets = train_data.get_batch_data(batch)
            # 　生成供tensorflow训练用的数据
            feed = {model.inputs: batch_inputs, model.targets: batch_targets, model.initial_state_fw: new_state_fw, model.initial_state_bw: new_state_bw}
            batch_loss, new_state_fw, new_state_bw, _ = sess.run([model.loss, model.final_state_fw, model.final_state_bw, model.optimizer], feed_dict=feed)
            total_loss += batch_loss
            step += 1
            if step % 100 == 0:
                saver.save(sess, './comments_model/bi_LSTM/', global_step=step)
                if step % 10000 == 0:
                    print(step)
        print('Train Loss at step {}: {:5.6f}'.format(index+1, total_loss / num_batches))

#%% ===================================生成句子 ============================= 
def pick_top_n(preds, vocab_size, top_n=10):
    # 从预测结果中选取前top_n个最可能的字符
    p = np.squeeze(preds)
    #p1 = list(p)
    #c = p1.index(max(p1))
    p = p[1:]
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size-1, 1, p=p)[0]+1
    return c
        
def sample(n_words, vocab_size, batch_size, num_batches, prime):
    #prime: 起始文本    
    samples=[prime]
    # sampling=True意味着batch的size=1 x 1
    model = Bi_LSTM(vocab_size, batch_size, num_batches, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        checkpoint_file = tf.train.latest_checkpoint('./comments_model/bi_LSTM')
        saver.restore(sess, checkpoint_file)
        new_state_fw, new_state_bw = sess.run([model.initial_state_fw, model.initial_state_bw])        
        # 不断生成字符，直到达到指定数目
        c = word2id.get(prime)
        for i in range(n_words):
            test_word_id = c
            if test_word_id == word2id.get('EOS'):
                break
            else:
                feed = {model.inputs: [[test_word_id]],
                        model.initial_state_fw: new_state_fw, model.initial_state_bw: new_state_bw}
                #preds= sess.run([model.prediction], feed_dict=feed)
                preds, new_state_fw, new_state_bw = sess.run([model.prediction, model.final_state_fw, model.final_state_bw], feed_dict=feed)
                c = pick_top_n(preds, vocab_size)
                samples.extend(x for x,v in word2id.items() if v==c)      
    print(' '.join(samples))
    
for i in range(5):  
    #sample(10, vocab_size, batch_size, num_batches, prime = "you")
    for j in ["thank", "beautiful", "very", "bro", "this","you" ]:
        sample(20, vocab_size, batch_size, num_batches, prime = j)

