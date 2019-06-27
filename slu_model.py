import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class slu_model(object):
    def __init__(self, max_seq_len, intent_dim, args):
        self.hidden_size = 128
        self.intent_dim = intent_dim # one hot encoding
        self.conv_size = 3
        self.num_dist = 59
        self.num_talker = 2 # {"Tourist", "Guide"}
        self.embedding_dim = 200 # read from glove
        self.total_word = 400001 # total word embedding vectors
        self.max_seq_len = max_seq_len
        self.hist_len = 7
        self.add_variables()
        self.add_placeholders()
        self.add_variables()
        self.build_graph(args)
        self.add_loss()
        self.add_train_op()
        self.init_embedding()
        self.init_model = tf.global_variables_initializer()

    def init_embedding(self):
        self.init_embedding = self.embedding_matrix.assign(self.read_embedding_matrix)

    def add_variables(self):
        self.embedding_matrix = tf.Variable(tf.truncated_normal([self.total_word, self.embedding_dim]), dtype=tf.float32, name="glove_embedding")
        self.dist_embedding_matrix = tf.Variable(tf.truncated_normal([self.num_dist, self.hidden_size]), dtype=tf.float32, name="dist_embedding")
        self.talker_embedding_matrix = tf.Variable(tf.truncated_normal([self.num_talker, self.hidden_size]), dtype=tf.float32, name="talker_embedding")

    def add_placeholders(self):
        self.history_intent = tf.placeholder(tf.float32, [None, self.hist_len * 2, self.intent_dim])
        self.tourist_input_intent, self.guide_input_intent = tf.split(self.history_intent, num_or_size_splits=2, axis=1)
        self.history_distance = tf.placeholder(tf.int32, [None, self.hist_len * 2])
        self.tourist_dist, self.guide_dist = tf.split(self.history_distance, num_or_size_splits=2, axis=1)
        self.read_embedding_matrix = tf.placeholder(tf.float32, [self.total_word, self.embedding_dim])
        self.labels = tf.placeholder(tf.float32, [None, self.intent_dim])
        self.current_nl_len = tf.placeholder(tf.int32, [None])
        self.current_nl = tf.placeholder(tf.int32, [None, self.max_seq_len])

        self.history_nl_len = tf.placeholder(tf.int32, [None, self.hist_len * 2])
        self.history_nl = tf.placeholder(tf.int32, [None, self.hist_len * 2, self.max_seq_len])
        self.current_talker = tf.placeholder(tf.int32, [None])
    
    def nl_biRNN(self, args):
        with tf.variable_scope("current_nl"):
            wemb = tf.nn.embedding_lookup(self.embedding_matrix, self.current_nl)
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            all_states, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, wemb, sequence_length=self.current_nl_len, dtype=tf.float32)
            first_all_states = all_states[0]+all_states[1]
            nl_outputs = final_states[0].h+final_states[1].h

        nl_outputs_dense_exp = tf.expand_dims(nl_outputs,axis=1)
        history_intent_dense = tf.layers.dense(inputs=self.history_intent, units=self.hidden_size, use_bias=False, kernel_initializer=tf.random_normal_initializer)
        dist_embedding = tf.nn.embedding_lookup(self.dist_embedding_matrix, self.history_distance)
        if args.level=='role':
            tourist_history_intent_dense, guide_history_intent_dense = tf.split(history_intent_dense, num_or_size_splits=2, axis=1)
            tourist_dist_embedding, guide_dist_embedding = tf.split(dist_embedding, num_or_size_splits=2, axis=1)

            tourist_hist_b = tf.Variable(tf.constant(0.0, shape=[self.hidden_size]))
            tourist_hist_w = tf.Variable(tf.truncated_normal([1,1,self.hidden_size],stddev=0.1))
            tourist_hist_dist_b = tf.Variable(tf.constant(0.0, shape=[self.hidden_size]))
            tourist_hist_dist_w = tf.Variable(tf.truncated_normal([1,1,self.hidden_size],stddev=0.1))

            if args.use_distinct_w == 'yes':
                guide_hist_b = tf.Variable(tf.constant(0.0, shape=[self.hidden_size]))
                guide_hist_w = tf.Variable(tf.truncated_normal([1,1,self.hidden_size],stddev=0.1))
                guide_hist_dist_b = tf.Variable(tf.constant(0.0, shape=[self.hidden_size]))
                guide_hist_dist_w = tf.Variable(tf.truncated_normal([1,1,self.hidden_size],stddev=0.1))
            elif args.use_distinct_w == 'no':
                guide_hist_b = tourist_hist_b
                guide_hist_w = tourist_hist_w
                guide_hist_dist_b = tourist_hist_dist_b
                guide_hist_dist_w = tourist_hist_dist_w
            
            if args.talker_applied_to:
                talker_embedding = tf.nn.embedding_lookup(self.talker_embedding_matrix, self.current_talker)
                talker_embedding_exp = tf.expand_dims(talker_embedding,axis=1)
                
                if 'Intent' in args.talker_applied_to:
                    tourist_hist_alphas = tf.reduce_sum(tf.multiply(tourist_hist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(tf.add(tourist_history_intent_dense, nl_outputs_dense_exp),talker_embedding_exp),tourist_hist_b))),axis=2)
                    guide_hist_alphas = tf.reduce_sum(tf.multiply(guide_hist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(tf.add(guide_history_intent_dense, nl_outputs_dense_exp),talker_embedding_exp),guide_hist_b))),axis=2)
                else:
                    tourist_hist_alphas = tf.reduce_sum(tf.multiply(tourist_hist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(tourist_history_intent_dense, nl_outputs_dense_exp),tourist_hist_b))),axis=2)
                    guide_hist_alphas = tf.reduce_sum(tf.multiply(guide_hist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(guide_history_intent_dense, nl_outputs_dense_exp),guide_hist_b))),axis=2)
                if 'Dist' in args.talker_applied_to:
                    tourist_hist_dist_alphas = tf.reduce_sum(tf.multiply(tourist_hist_dist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(tf.add(nl_outputs_dense_exp,tourist_dist_embedding),talker_embedding_exp),tourist_hist_dist_b))),axis=2)
                    guide_hist_dist_alphas = tf.reduce_sum(tf.multiply(guide_hist_dist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(tf.add(nl_outputs_dense_exp,guide_dist_embedding),talker_embedding_exp),guide_hist_dist_b))),axis=2)
                else:
                    tourist_hist_dist_alphas = tf.reduce_sum(tf.multiply(tourist_hist_dist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(nl_outputs_dense_exp,tourist_dist_embedding),tourist_hist_dist_b))),axis=2)
                    guide_hist_dist_alphas = tf.reduce_sum(tf.multiply(guide_hist_dist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(nl_outputs_dense_exp,guide_dist_embedding),guide_hist_dist_b))),axis=2)

            else:
                tourist_hist_alphas = tf.reduce_sum(tf.multiply(tourist_hist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(tourist_history_intent_dense, nl_outputs_dense_exp),tourist_hist_b))),axis=2)
                guide_hist_alphas = tf.reduce_sum(tf.multiply(guide_hist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(guide_history_intent_dense, nl_outputs_dense_exp),guide_hist_b))),axis=2)
            
                tourist_hist_dist_alphas = tf.reduce_sum(tf.multiply(tourist_hist_dist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(nl_outputs_dense_exp,tourist_dist_embedding),tourist_hist_dist_b))),axis=2)
                guide_hist_dist_alphas = tf.reduce_sum(tf.multiply(guide_hist_dist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(nl_outputs_dense_exp,guide_dist_embedding),guide_hist_dist_b))),axis=2)
            
            tourist_hist_att = tf.nn.softmax(tourist_hist_alphas)
            guide_hist_att = tf.nn.softmax(guide_hist_alphas)
            
            tourist_hist_dist_att = tf.nn.softmax(tourist_hist_dist_alphas)
            guide_hist_dist_att = tf.nn.softmax(guide_hist_dist_alphas)
            tourist_hist_att_exp = tf.expand_dims(tourist_hist_att,axis=2)
            guide_hist_att_exp = tf.expand_dims(guide_hist_att,axis=2)

            tourist_hist_dist_att_exp = tf.expand_dims(tourist_hist_dist_att,axis=2)
            guide_hist_dist_att_exp = tf.expand_dims(guide_hist_dist_att,axis=2)

            tourist_history_intent_dense, guide_history_intent_dense = tf.split(history_intent_dense, num_or_size_splits=2, axis=1)

            tourist_dist_embedding, guide_dist_embedding = tf.split(dist_embedding, num_or_size_splits=2, axis=1)

            tourist_aligned_hist_fr_intent = tf.reduce_sum(tf.multiply(tourist_hist_att_exp,tourist_history_intent_dense),axis=1)
            guide_aligned_hist_fr_intent = tf.reduce_sum(tf.multiply(guide_hist_att_exp,guide_history_intent_dense),axis=1)

            tourist_aligned_dist_fr_intent = tf.reduce_sum(tf.multiply(tourist_hist_att_exp,tourist_dist_embedding),axis=1)
            guide_aligned_dist_fr_intent = tf.reduce_sum(tf.multiply(guide_hist_att_exp,guide_dist_embedding),axis=1)

            tourist_aligned_hist_fr_dist = tf.reduce_sum(tf.multiply(tourist_hist_dist_att_exp,tourist_history_intent_dense),axis=1)
            guide_aligned_hist_fr_dist = tf.reduce_sum(tf.multiply(guide_hist_dist_att_exp,guide_history_intent_dense),axis=1)

            tourist_aligned_dist_fr_dist = tf.reduce_sum(tf.multiply(tourist_hist_dist_att_exp,tourist_dist_embedding),axis=1)
            guide_aligned_dist_fr_dist = tf.reduce_sum(tf.multiply(guide_hist_dist_att_exp,guide_dist_embedding),axis=1)

            if 'Intent' in args.att_out:
                if 'Intent' in args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, tourist_aligned_hist_fr_intent, guide_aligned_hist_fr_intent], axis=1)
                    except NameError:
                        history_summary = tf.concat([tourist_aligned_hist_fr_intent, guide_aligned_hist_fr_intent], axis=1)
                if 'Dist' in args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, tourist_aligned_hist_fr_dist, guide_aligned_hist_fr_dist], axis=1)
                    except NameError:
                        history_summary = tf.concat([tourist_aligned_hist_fr_dist, guide_aligned_hist_fr_dist], axis=1)
                if not args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, tf.reduce_sum(tourist_history_intent_dense, axis=1), tf.reduce_sum(guide_history_intent_dense, axis=1)], axis=1)
                    except NameError:
                        history_summary = tf.concat([tf.reduce_sum(tourist_history_intent_dense, axis=1), tf.reduce_sum(guide_history_intent_dense, axis=1)], axis=1)

            if 'Dist' in args.att_out:
                if 'Intent' in args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, tourist_aligned_dist_fr_intent, guide_aligned_dist_fr_intent], axis=1)
                    except NameError:
                        history_summary = tf.concat([tourist_aligned_dist_fr_intent, guide_aligned_dist_fr_intent], axis=1)
                if 'Dist' in args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, tourist_aligned_dist_fr_dist, guide_aligned_dist_fr_dist], axis=1)
                    except NameError:
                        history_summary = tf.concat([tourist_aligned_dist_fr_dist, guide_aligned_dist_fr_dist], axis=1)
                if not args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, tf.reduce_sum(tourist_dist_embedding, axis=1), tf.reduce_sum(guide_dist_embedding, axis=1)], axis=1)
                    except NameError:
                        history_summary = tf.concat([tf.reduce_sum(tourist_dist_embedding, axis=1), tf.reduce_sum(guide_dist_embedding, axis=1)], axis=1)

        elif args.level=='sentence':
            hist_b = tf.Variable(tf.constant(0.0, shape=[self.hidden_size]))
            hist_w = tf.Variable(tf.truncated_normal([1,1,self.hidden_size],stddev=0.1))
            hist_dist_b = tf.Variable(tf.constant(0.0, shape=[self.hidden_size]))
            hist_dist_w = tf.Variable(tf.truncated_normal([1,1,self.hidden_size],stddev=0.1))
            
            if args.talker_applied_to:
                talker_embedding = tf.nn.embedding_lookup(self.talker_embedding_matrix, self.current_talker)
                talker_embedding_exp = tf.expand_dims(talker_embedding,axis=1)
                if 'Intent' in args.talker_applied_to:
                    hist_alphas = tf.reduce_sum(tf.multiply(hist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(tf.add(history_intent_dense, nl_outputs_dense_exp),talker_embedding_exp),hist_b))),axis=2)
                else:
                    hist_alphas = tf.reduce_sum(tf.multiply(hist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(history_intent_dense, nl_outputs_dense_exp),hist_b))),axis=2)
                if 'Dist' in args.talker_applied_to:
                    hist_dist_alphas = tf.reduce_sum(tf.multiply(hist_dist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(tf.add(nl_outputs_dense_exp,dist_embedding),talker_embedding_exp),hist_dist_b))),axis=2)
                else:
                    hist_dist_alphas = tf.reduce_sum(tf.multiply(hist_dist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(nl_outputs_dense_exp,dist_embedding),hist_dist_b))),axis=2)

            else:
                hist_alphas = tf.reduce_sum(tf.multiply(hist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(history_intent_dense, nl_outputs_dense_exp),hist_b))),axis=2)
                hist_dist_alphas = tf.reduce_sum(tf.multiply(hist_dist_w,tf.nn.tanh(tf.nn.bias_add(tf.add(nl_outputs_dense_exp,dist_embedding),hist_dist_b))),axis=2)

            hist_att = tf.nn.softmax(hist_alphas)
            hist_dist_att = tf.nn.softmax(hist_dist_alphas)
            
            hist_att_exp = tf.expand_dims(hist_att,axis=2)
            hist_dist_att_exp = tf.expand_dims(hist_dist_att,axis=2)
            
            aligned_hist_fr_intent = tf.reduce_sum(tf.multiply(hist_att_exp,history_intent_dense),axis=1)
            aligned_dist_fr_intent = tf.reduce_sum(tf.multiply(hist_att_exp,dist_embedding),axis=1)
            aligned_hist_fr_dist = tf.reduce_sum(tf.multiply(hist_dist_att_exp,history_intent_dense),axis=1)
            aligned_dist_fr_dist = tf.reduce_sum(tf.multiply(hist_dist_att_exp,dist_embedding),axis=1)

            if 'Intent' in args.att_out:
                if 'Intent' in args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, aligned_hist_fr_intent], axis=1)
                    except NameError:
                        history_summary = aligned_hist_fr_intent
                if 'Dist' in args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, aligned_hist_fr_dist], axis=1)
                    except NameError:
                        history_summary = aligned_hist_fr_dist
                if not args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, tf.reduce_sum(history_intent_dense, axis=1)], axis=1)
                    except NameError:
                        history_summary = tf.reduce_sum(history_intent_dense, axis=1)

            if 'Dist' in args.att_out:
                if 'Intent' in args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, aligned_dist_fr_intent], axis=1)
                    except NameError:
                        history_summary = aligned_dist_fr_intent
                if 'Dist' in args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, aligned_dist_fr_dist], axis=1)
                    except NameError:
                        history_summary = aligned_dist_fr_dist
                if not args.att_to:
                    try:
                        history_summary = tf.concat([history_summary, tf.reduce_sum(dist_embedding, axis=1)], axis=1)
                    except NameError:
                        history_summary = tf.reduce_sum(dist_embedding, axis=1)

        with tf.variable_scope("second_current_nl"):
            history_summary = tf.expand_dims(history_summary, axis=1)
            replicate_summary = tf.tile(history_summary, [1, self.max_seq_len, 1])
            concat_input = tf.concat([first_all_states, replicate_summary], axis=2)
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, concat_input, sequence_length=self.current_nl_len, dtype=tf.float32)
            outputs = tf.concat([final_states[0].h,final_states[1].h], axis=1)
        
        return outputs

    def build_graph(self, args):
        nl_outputs = self.nl_biRNN(args)
        self.output = tf.layers.dense(inputs=nl_outputs, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
        self.intent_output = tf.sigmoid(self.output)

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.output))
        
    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.loss)
