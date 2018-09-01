# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 14:52:16 2018

@author: 47532
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import random

from tqdm import tqdm
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

batch_size = 1024
embed_size = 32
epochs = 2
        
uid = []
iid = []
click = []
for i in range(100):
    i = str(i).zfill(2)
    for line in open("part-000" + i): 
        rec = line.strip().split(' ')
        uid.append(rec[0])
        iid.append(rec[1])
        click.append(rec[4])
        
trend = pd.DataFrame({'uid':uid, 'iid': iid, 'click': click},columns = ['uid','iid', 'click'])

x = trend.assign(**{'iid':trend['iid'].str.split(','), 'click':trend['click'].str.split(',')})
trend1 = pd.DataFrame({'uid':np.repeat(x['uid'].values, x['iid'].str.len())}).assign(**{'iid':np.concatenate(x['iid'].values),'click':np.concatenate(x['click'].values)})[x.columns.tolist()]
trend1['iid'] = trend1['iid'].astype(int)
trend1['click'] = trend1['click'].astype(int)
# pos_n = len(trend1[trend1['click'] == '1'])
# neg_n = len(trend1) - pos_n

trend_pos = trend1[trend1['click'] == 1]
trend_neg = trend1[trend1['click'] == 0]

uid_count = trend_pos['uid'].value_counts()
uid_freq = uid_count[uid_count > 10].index.tolist()

trend_pos_freq = trend_pos[trend_pos['uid'].isin(uid_freq)]
trend_pos_sparse = trend_pos[~trend_pos['uid'].isin(uid_freq)]

trend_neg_freq = trend_neg[trend_neg['uid'].isin(uid_freq)]

test_neg = trend_neg_freq.sample(frac = 0.015,replace = False)
train,test = train_test_split(trend_pos_freq, test_size=0.1)
train_trend = pd.concat([trend_pos_sparse,train],axis = 0)
test_trend = pd.concat([test_neg,test],axis = 0)

uid_unique = train_trend['uid'].unique()
uid_label = preprocessing.LabelEncoder().fit(uid_unique)
train_trend['label'] = uid_label.transform(train_trend['uid'])
test_trend['label'] = uid_label.transform(test_trend['uid'])
test_trend['iid'] = test_trend['iid'] - 1
train_trend['iid'] = train_trend['iid'] - 1

# Check if test set has uid that not in train set
#train_uid = set(train_trend['uid'].unique())
#len(train_uid)
#test_uid = set(test_trend['uid'].unique() )
#len(test_uid)
#verify = train_uid.intersection(test_uid)  
#len(verify)



class TrainData:
    def __init__(self, users, preferred_items):
        self.users = users
        self.preferred_items = preferred_items
        assert len(users) == len(preferred_items)
        self.n = len(users)
        self.u_i_set = {}
        self.max_item_id = -1
        for i in range(self.n):
            uid = self.users[i]
            iid = self.preferred_items[i]
            self.max_item_id = max(self.max_item_id, iid)
            if uid not in self.u_i_set:
                self.u_i_set[uid] = {}
            self.u_i_set[uid][iid] = True

    def get_data(self, batch):
        global batch_size
        start_pos = batch * batch_size
        end_pos = min((batch + 1) * batch_size, self.n)
        batch_other_items = list()
        for i in range(start_pos, end_pos):
            uid = self.users[i]
            while True:
                x = random.randint(0, self.max_item_id)
                if x not in self.u_i_set[uid]:
                    batch_other_items.append(x)
                    break
        return self.users[start_pos:end_pos], self.preferred_items[start_pos:end_pos], batch_other_items

    def get_num_batches(self):
        return max(self.n - 1, 0) // batch_size + 1
    
def get_data(date):
    users_num = len(train_trend['uid'].unique())
    items_num = max(trend1['iid'])  
    users = train_trend['label'].get_values().tolist()
    preferred_items = train_trend['iid'].get_values().tolist()
    train_data = TrainData(users, preferred_items)
    del users
    del preferred_items
    return train_data, users_num, items_num


get_data("20180705")


class Model:
    def __init__(self, user_size, item_size, embed_size, num_batches, lambd, learning_rate):
        self.graph = tf.Graph()
        self.user_size = user_size
        self.item_size = item_size
        self.embed_size = embed_size
        self.num_batches = num_batches
        self.lambd = lambd
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.build_graph()

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.train_users = tf.placeholder(tf.int32, name='train_users')
            self.train_preferred_items = tf.placeholder(tf.int32, name='train_preferred_items')
            self.train_other_items = tf.placeholder(tf.int32, name='train_other_items')
            self.evaluation_users = tf.placeholder(tf.int32, name='evaluation_users')
            self.evaluation_items = tf.placeholder(tf.int32, name='evaluation_items')
            self.evaluation_scores = tf.placeholder(tf.float32, name='evaluation_scores')

    def _create_embedding(self):
        with tf.name_scope("embed"):          
            self.U = tf.get_variable("U", shape=[self.user_size, self.embed_size],
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd / self.num_batches),
                                     initializer=tf.contrib.layers.xavier_initializer(seed=1))
            self.I = tf.get_variable("I", shape=[self.item_size, self.embed_size],
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd / self.num_batches),
                                     initializer=tf.contrib.layers.xavier_initializer(seed=1))

    def _create_loss(self):
        with tf.name_scope("loss"):
            user_embed = tf.nn.embedding_lookup(self.U, self.train_users)
            reshaped_user_embed = tf.reshape(user_embed, [-1, 1, self.embed_size])

            preferred_item_embed = tf.nn.embedding_lookup(self.I, self.train_preferred_items)
            reshaped_preferred_item_embed = tf.reshape(preferred_item_embed, [-1, self.embed_size, 1])  # transposed

            other_item_embed = tf.nn.embedding_lookup(self.I, self.train_other_items)
            reshaped_other_item_embed = tf.reshape(other_item_embed, [-1, self.embed_size, 1])  # transposed

            preferred_score = tf.squeeze(tf.matmul(reshaped_user_embed, reshaped_preferred_item_embed))
            other_score = tf.squeeze(tf.matmul(reshaped_user_embed, reshaped_other_item_embed))
            diff = preferred_score - other_score

            self.diff = tf.reduce_mean(diff, name='diff')
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            self.loss_with_reg = tf.add(tf.reduce_sum(- tf.sigmoid(diff)), self.reg_loss, name='loss_with_reg')

    def _create_evaluation(self):
        with tf.name_scope("evaluation"):
            
            user_embed = tf.nn.embedding_lookup(self.U, self.evaluation_users)
            item_embed = tf.nn.embedding_lookup(self.I, self.evaluation_items)
            reshaped_user_embed = tf.reshape(user_embed, [-1, 1, self.embed_size])
            reshaped_item_embed = tf.reshape(item_embed, [-1, self.embed_size, 1])
            self.scores_hat = tf.sigmoid(tf.squeeze(tf.matmul(reshaped_user_embed, reshaped_item_embed))
                                         , name='scores_hat')
            diff = self.evaluation_scores - self.scores_hat
            self.l2_dis = tf.reduce_mean(diff ** 2, name='l2_dis')

    def _predict_mat(self):
        with tf.name_scope('predict'):
            self.W = tf.sigmoid(tf.matmul(self.U, tf.transpose(self.I), name='W'))

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
                self._predict_mat()

    def train(self, train_data, date, num_train_steps=10):
        os.system('rm -rf /data/sing_model/%s' % date)
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('/data/sing_model/%s/bpr_board' % date, sess.graph)
            initial_step = 0  # self.global_step.eval(session=sess)
            step = 0
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            for index in range(initial_step, initial_step + num_train_steps):
                total_diff = 0.0
                total_loss_with_reg = 0.0
                for batch in tqdm(range(self.num_batches)):
                    batch_users, batch_items, other_items = train_data.get_data(batch)
                    feed_dict = {self.train_users: batch_users,
                                 self.train_preferred_items: batch_items,
                                 self.train_other_items: other_items}
                    diff_batch, loss_with_reg_batch, _, summary = sess.run([self.diff, self.loss_with_reg,
                                                                            self.optimizer, self.summary_op],
                                                                           feed_dict=feed_dict)
                    total_diff += diff_batch
                    total_loss_with_reg += loss_with_reg_batch
                    step += 1
                    if step % 100 == 0:
                        writer.add_summary(summary, global_step=step)
                        saver.save(sess, '/data/sing_model/%s/bpr/' % date, global_step=step)
                        if step % 10000 == 0:
                            print(step)
                print('Train Loss at step {}: {:5.6f}, {:5.6f}'.format(index,
                                                                       total_diff / train_data.get_num_batches(),
                                                                       total_loss_with_reg))

class TestData:
    def __init__(self, users, items, scores):
        self.users = users
        self.items = items
        self.scores = scores
        self.n = len(users)
        assert len(users) == len(items)
        assert len(users) == len(scores)

    def get_data(self, batch):
        global batch_size
        start_pos = batch * batch_size
        end_pos = min((batch + 1) * batch_size, self.n)
        return self.users[start_pos:end_pos], self.items[start_pos:end_pos], self.scores[start_pos:end_pos]

    def get_num_batches(self):
        return max(self.n - 1, 0) // batch_size + 1

def train_model(date):
    train, users_num, items_num = get_data("20180705") 
    print('Train size: %s' % train.n)
    print('Users: %s' % users_num)
    print('Items: %s' % items_num)

    model = Model(users_num, items_num, embed_size, train.get_num_batches(), 0.0001, 0.01)
    model.train(train, date, epochs)

train_model("20180705")

def predict():
    sess = tf.Session()
    checkpoint_file = tf.train.latest_checkpoint('/data/sing_model/20180705/bpr')
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file), clear_devices=True)
    saver.restore(sess, checkpoint_file)
    graph = sess.graph
    g_users = graph.get_operation_by_name("data/evaluation_users").outputs[0]
    g_items = graph.get_operation_by_name("data/evaluation_items").outputs[0]
    g_scores = graph.get_operation_by_name("data/evaluation_scores").outputs[0]

    g_scores_hat = graph.get_operation_by_name("evaluation/scores_hat").outputs[0]
    test_data = TestData(test_trend['label'], test_trend['iid'], test_trend['click'])
    scores_hat = []
    for batch in range(test_data.get_num_batches()):
        batch_users, batch_items, batch_scores = test_data.get_data(batch)
        feed_dict = {g_users: batch_users,
                     g_items: batch_items,
                     g_scores: batch_scores}
        batch_scores_hat = sess.run(g_scores_hat, feed_dict=feed_dict)
        scores_hat.extend(batch_scores_hat)
    from sklearn import metrics
    metrics.roc_auc_score(test_trend['click'], scores_hat)
    
