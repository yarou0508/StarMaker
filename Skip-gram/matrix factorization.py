# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:16:24 2018

@author: 47532
"""

import re
import regex
import tensorflow as tf
import os
from tqdm import tqdm
import pandas as pd
import random

def get_data(win_len):
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
                words.extend(word)
                sentences.append(word)   
            else:
                continue    
    words_sort = pd.DataFrame(words)[0].value_counts()                           
    word_bank = list(words_sort.index)
    word_bank.remove("'")
    word2id = {}  # word => id 的映射
    for i in range(len(word_bank)):
        word2id[word_bank[i]] = i
    inputs = []
    labels = []
    for sent in sentences:  # 输入是多个句子，这里每个循环处理一个句子
        for i in range(sent.__len__()):  # 处理单个句子中的每个单词
            start = max(0, i - win_len)  # 窗口为 [-win_len,+win_len],总计长2*win_len+1
            end = min(sent.__len__(), i + win_len + 1)
            # 将某个单词对应窗口中的其他单词转化为id计入label，该单词本身计入input
            for index in range(start, end):
                if index == i:
                    continue
                else:
                    input_id = word2id.get(sent[i])
                    label_id = word2id.get(sent[index])
                    if not (input_id and label_id):  # 如果单词不在词典中，则跳过
                        continue
                    inputs.append(input_id)
                    labels.append(label_id)
    return word_bank, len(word_bank), inputs, labels

class TrainData:
    def __init__(self, inputs, labels, word_bank, vocab_size):
        self.word_bank = word_bank
        self.vocab_size = vocab_size
        self.inputs = inputs
        self.labels = labels
        self.n = len(inputs)
        self.w_p_set = {}
        self.max_p_id = -1
        for i in range(self.n):
            cid = self.inputs[i]
            pid = self.labels[i]
            self.max_p_id = max(self.max_p_id,pid)
            if cid not in self.w_p_set:
                self.w_p_set[cid] = {}
            self.w_p_set[cid][pid] = True
    def get_batch_data(self, batch):
        global batch_size
        start_pos = batch * batch_size
        end_pos = min((batch + 1) * batch_size, self.n)
        batch_negatives = list()
        for i in range(start_pos, end_pos):
            cid = self.inputs[i]
            while True:
                x = random.randint(0, self.max_p_id)
                if x not in self.w_p_set[cid]:
                    batch_negatives.append(x)
                    break
        return self.inputs[start_pos:end_pos], self.labels[start_pos:end_pos],batch_negatives 
    def get_num_batches(self):
        return max(self.n - 1, 0) // batch_size + 1

class Model:
    def __init__(self, vocab_size, embedding_size, num_batches, lambd,learning_rate):
        self.graph = tf.Graph()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_batches = num_batches
        self.lambd = lambd
        self.batch_size = None
        self.lr = learning_rate
        self.build_graph()
    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size],name = 'train_inputs')
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size], name = 'train_labels')
            self.train_negatives = tf.placeholder(tf.int32, shape=[self.batch_size], name='train_negatives')
            self.test_word_id = tf.placeholder(tf.int32, shape=[None], name = 'test_word_id')
    def _create_embedding(self):
        with tf.name_scope("embed"):
            self.embedding_dict = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size],
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd / self.num_batches),
                                     initializer=tf.random_uniform_initializer(-1,1,seed=1))
            #tf.Variable(tf.random_uniform((self.vocab_size, self.embedding_size), -1, 1, name = "embedding"))
            # 模型内部参数矩阵，初始为截断正太分布
            self.nce_weight = tf.get_variable('weight', shape = [self.vocab_size, self.embedding_size], 
                                              regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd / self.num_batches), 
                                              initializer = tf.random_uniform_initializer(-1,1,seed=1))
            #tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],stddev=1.0 / math.sqrt(self.embedding_size)),name = "weight")         
    def _create_loss(self):
        with tf.name_scope("loss"):
            input_embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs)
            reshaped_input_embed = tf.reshape(input_embed, [-1, 1, self.embedding_size])
            label_embed = tf.nn.embedding_lookup(self.nce_weight, self.train_labels)
            reshaped_label_embed = tf.reshape(label_embed, [-1, self.embedding_size, 1])
            negative_embed = tf.nn.embedding_lookup(self.nce_weight, self.train_negatives)
            reshaped_negative_embed = tf.reshape(negative_embed, [-1, self.embedding_size, 1]) 
            positive_score =  tf.squeeze(tf.matmul(reshaped_input_embed, reshaped_label_embed))
            negative_score = tf.squeeze(tf.matmul(reshaped_input_embed, reshaped_negative_embed))
            diff = positive_score - negative_score
            
            self.diff = tf.reduce_mean(diff, name='diff')
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            self.loss_with_reg = tf.add(tf.reduce_sum(- tf.sigmoid(diff)), self.reg_loss, name='loss_with_reg')
    def _create_evaluation(self):
        with tf.name_scope("evaluation"):           
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_dict), 1, keepdims=True))
            self.normed_embedding = self.embedding_dict / norm
            test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
            self.similarity = tf.matmul(test_embed, tf.transpose(self.normed_embedding), name = 'similarity')
    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_with_reg)
    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("diff", self.diff)
            tf.summary.scalar("loss_with_reg", self.loss_with_reg)
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
    def train(self, train_data,num_train_steps=10):
        os.system('rm -rf ./comments_model')
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('./comments_model/bpr', sess.graph)
            initial_step = 0  # self.global_step.eval(session=sess)
            step = 0
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            for index in range(initial_step, initial_step + num_train_steps):
                total_diff = 0.0
                total_loss_with_reg = 0.0
                for batch in tqdm(range(self.num_batches)):
                    batch_inputs,batch_labels,batch_negatives = train_data.get_batch_data(batch)
                    # 　生成供tensorflow训练用的数据
                    feed_dict = {self.train_inputs: batch_inputs,
                        self.train_labels: batch_labels,
                        self.train_negatives: batch_negatives}
                    diff_batch, loss_with_reg_batch, _, summary = sess.run([self.diff, self.loss_with_reg, self.optimizer, self.summary_op],feed_dict=feed_dict)
                    total_diff += diff_batch
                    total_loss_with_reg += loss_with_reg_batch
                    step += 1
                    if step % 100 == 0:
                        writer.add_summary(summary, global_step=step)
                        saver.save(sess, './comments_model/bpr/', global_step=step)
                        if step % 10000 == 0:
                            print(step)
                print('Train Loss at step {}: {:5.6f}, {:5.6f}'.format(index, total_diff / train_data.get_num_batches(),
                                                                       total_loss_with_reg / train_data.get_num_batches()))


word_bank, vocab_size, inputs, labels = get_data(4)
batch_size = 10240                
train_data = TrainData(inputs, labels, word_bank, vocab_size) # win_len, inputs, word_bank, vocab_size
print('Train size: %s' % train_data.n)
print('Vocab_size: %s' % vocab_size)                
model = Model(vocab_size, 32, train_data.get_num_batches(), 0.01, 0.01)    # vocab_size, embedding_size, num_batches, lambd, num_sampled, learning_rate           
model.train(train_data, 15) # train_data, epochs
    
    
def predict(test_words):
    sess = tf.Session()
    checkpoint_file = tf.train.latest_checkpoint('./comments_model/bpr')
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file), clear_devices=True)
    saver.restore(sess, checkpoint_file)
    graph = sess.graph
    g_test_word_id = graph.get_operation_by_name("data/test_word_id").outputs[0]
    g_similarity = graph.get_operation_by_name("evaluation/similarity").outputs[0] 
    word2id = {}  # word => id 的映射
    for i in range(vocab_size):
        word2id[word_bank[i]] = i        
    test_word_id = [word2id.get(x) for x in test_words]
    feed_dict = {g_test_word_id: test_word_id}
    similarity = sess.run(g_similarity, feed_dict=feed_dict)
    top_k = 8
    for i in range(len(test_words)):
        nearest = (-similarity[i, :]).argsort()[1:top_k+1]        
        log = 'Nearest to %s:' % test_words[i]
        for k in range(top_k):
            close_word = [x for x,v in word2id.items() if v == nearest[k]]
            log = '%s %s,' % (log, close_word)
        print(log)
test_words = ['nice', 'perfect' ,'song', 'thnx', '😘', 'voice', '😡','dear']

predict(test_words)


