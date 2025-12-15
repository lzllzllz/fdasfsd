'''
Reference: DR-CFR in ICLR(2020)
[1] Hassanpour N, Greiner R. Learning disentangled representations for counterfactual regression[C]//International Conference on Learning Representations. 2020.

'''
import tensorflow as tf
import numpy as np
from functions import utils

class Net(object):
    def __init__(self, n, x_dim, y_dim, t_dim, FLAGS):
        self.n = n
        self.x_dim = x_dim
        self.t_dim = t_dim 
        self.y_dim = y_dim
        self.FLAGS = FLAGS

        self.wd_loss = 0

        self.x = tf.compat.v1.placeholder('float', shape=[None, x_dim], name='x')
        self.t = tf.compat.v1.placeholder('float', shape=[None, t_dim], name='t')
        self.y = tf.compat.v1.placeholder('float', shape=[None, y_dim], name='y')
        self.do_in = tf.compat.v1.placeholder("float", name='dropout_in')
        self.do_out = tf.compat.v1.placeholder("float", name='dropout_out')
        self.p_t = tf.compat.v1.placeholder("float", name='p_treated')##干预倾向得分
        self.t_threshold = tf.compat.v1.placeholder("float", name='treatment_threshold')
        self.y_threshold = tf.compat.v1.placeholder("float", name='outcome_threshold')
        self.I = tf.compat.v1.placeholder("int32", shape=[None, ], name='I')
        
        self.is_training = tf.compat.v1.placeholder(dtype = tf.bool)

        # learning rate 
        self.lr = FLAGS.lr
        self.global_step = tf.Variable(0, trainable=False)

        #个很直接的想法就是随着训练的进行，动态设置学习率——随着训练次数增加，学习率逐步减小。
        self.lr_exp = tf.compat.v1.train.exponential_decay(learning_rate = self.lr, global_step=self.global_step,
                                                           decay_steps = FLAGS.lr_decay_steps,
                                                           decay_rate = FLAGS.lr_decay_rate, staircase=True)
        self.boundaries = utils.string2list(FLAGS.lr_boundaries)
        self.values = utils.string2list(FLAGS.lr_values)
        self.lr_piecewise = tf.compat.v1.train.piecewise_constant(self.global_step, boundaries = self.boundaries, values = self.values) # 每运行一次，self.global_step+1
        
        # index partitions: i0(control) i1(treatment)
        if self.t_dim == 1:
            self.i0 = tf.cast(tf.where(self.t < self.t_threshold)[:, 0], dtype=tf.int32)
            self.i1 = tf.cast(tf.where(self.t >= self.t_threshold)[:, 0], dtype=tf.int32)
        elif self.t_dim >= 2:
            self.i0 = tf.cast(tf.where(self.t[:,0] >= 1)[:, 0], dtype=tf.int32)
            self.i1 = tf.cast(tf.where(self.t[:,0] <= 0)[:, 0], dtype=tf.int32) 
        self.partitions_i0_i1 = tf.reduce_sum(tf.one_hot(self.i1, depth = tf.shape(self.t)[0], dtype='int32'), 0)
 

        # activation
        if FLAGS.activation.lower() == 'elu':
            self.activation = tf.nn.elu
        elif FLAGS.activation.lower() == 'tanh':
            self.activation = tf.nn.tanh
        else:
            self.activation = tf.nn.relu
        # initializer
        self.initializer = tf.contrib.layers.xavier_initializer(seed=FLAGS.seed)

        self.build_graph()
        self.calculate_loss()
        self.setup_train_ops()

        self.record_in_tensorboard()
    
    def FC_layer(self, dim_in, dim_out, name, wd=0, b=True, weight_init = 0.01):
        bias = 0

        if self.FLAGS.var_from == 'get_variable':
            weight = tf.compat.v1.get_variable(name='weight' + name,
                                               shape=[dim_in, dim_out], initializer=self.initializer)
            if b:
                bias = tf.compat.v1.get_variable(name='bias' + name,
                                                 shape=[1, dim_out], initializer=tf.constant_initializer())
        else:
            weight = tf.Variable(tf.random.normal([dim_in, dim_out],
                                                  stddev=weight_init / np.sqrt(dim_in)), name='weight' + name)
            if b:
                bias = tf.Variable(tf.zeros([1, dim_out]), name='bias' + name)

        if wd>0:
            self.wd_loss += wd * tf.nn.l2_loss(weight)

        return weight, bias
    
    def build_graph(self):

        ''' Representation: A, I, C'''
        with tf.compat.v1.variable_scope('representation'):
            with tf.compat.v1.variable_scope('representation_I_'):
                # 工具变量表示
                self.rep_I, self.reps_I, self.w_I, self.b_I = self.representation(input=self.x,
                                                                                dim_in=self.x_dim,
                                                                                dim_out=self.FLAGS.rep_dim,
                                                                                layer=self.FLAGS.rep_layer,
                                                                                name='Instrument')
            with tf.compat.v1.variable_scope('representation_C_'):                           
                self.rep_C, self.reps_C, self.w_C, self.b_C = self.representation(input=self.x,
                                                                                dim_in=self.x_dim,
                                                                                dim_out=self.FLAGS.rep_dim,
                                                                                layer=self.FLAGS.rep_layer,
                                                                                name='Confounder')
            with tf.compat.v1.variable_scope('representation_A_'):
                self.rep_A, self.reps_A, self.w_A, self.b_A = self.representation(input=self.x,
                                                                                dim_in=self.x_dim,
                                                                                dim_out=self.FLAGS.rep_dim,
                                                                                layer=self.FLAGS.rep_layer,
                                                                                name='Adjustment')
        ''' Treatment: {C->T} {I,C->T} '''
        with tf.compat.v1.variable_scope('treatment'):
            with tf.compat.v1.variable_scope('treatment_C_'):
                self.mu_T_C, self.mus_T_C, self.w_muT_C, self.b_muT_C = self.treatment(
                    input=self.rep_C,
                    dim_in=self.FLAGS.rep_dim,
                    dim_out=self.FLAGS.net_t_dim,
                    layer=self.FLAGS.t_layer,
                    name='Mu_Treatment_C')
            with tf.compat.v1.variable_scope('treatment_CI_'):
                self.mu_T_CI, self.mus_T_CI, self.w_muT_CI, self.b_muT_CI = self.treatment(
                    input=tf.concat((self.rep_C,self.rep_A),axis=1),
                    dim_in=self.FLAGS.rep_dim * 2, 
                    dim_out=self.FLAGS.net_t_dim,
                    layer=self.FLAGS.t_layer,
                    name='Mu_Treatment_CI')

        ''' Outcome: {T+C+A -> Y} '''
        if self.t_dim == 2:
            with tf.compat.v1.variable_scope('outcome'):
                with tf.compat.v1.variable_scope('outcome_TCA_'):
                    self.mu_Y, self.mu_YCF, self.mus_Y, self.w_muY, self.b_muY = self.output(
                        input=tf.concat((self.rep_C, self.rep_A), axis=1),
                        dim_in=self.FLAGS.rep_dim * 2,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx')
        else:
            with tf.compat.v1.variable_scope('outcome'):
                with tf.compat.v1.variable_scope('outcome_TCA_'):
                    self.mu_Y, self.mu_YCF, self.mus_Y, self.w_muY, self.b_muY = self.output(
                        input=tf.concat((self.rep_C, self.rep_A, self.t), axis=1),
                        dim_in=self.FLAGS.rep_dim * 2 + self.t_dim,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx')
        
        ''' Sample Weight: used to weight the factual loss '''
        _, self.ps_score, _, _ = self.log_loss(self.mu_T_C, self.t)
        self.p_1 = tf.reduce_mean(self.t[:,1])
        self.p_0 = tf.reduce_mean(self.t[:,0])
        self.p_t_vec = tf.reshape(self.p_1 * self.t[:,1] + self.p_0 * (1-self.t[:,1]), shape=[-1,1])
        self.sample_weight = 1 + self.p_t_vec/(1-self.p_t_vec) * (1-self.ps_score)/self.ps_score


    ##损失函数
    def calculate_loss(self):
        
        # calculate w_I/C/A_mean and self.IPM_I/C/A
        self.ICA_W_setting()
        self.IPM()

        ''' loss for predict T'''
        if self.t_dim >= 2:
            self.t_pred_C, _, self.loss_C_T, self.acc_C_T = self.log_loss(self.mu_T_C, self.t)
            self.t_pred_CI, _, self.loss_CI_T, self.acc_CI_T = self.log_loss(self.mu_T_CI, self.t)
        else: 
            self.t_pred_C, self.t_pred_CI = self.mu_T_C, self.mu_T_CI
            self.loss_C_T, _ = self.l2_loss(self.mu_T_C, self.t)
            self.loss_CI_T, _ = self.l2_loss(self.mu_T_CI, self.t)

        # ''' loss for predict Y'''
        if self.y_dim >= 2:
            self.y_pred, _, self.loss_TCA_Y, _ = self.log_loss(self.mu_Y, self.y, True)
        else:
            self.y_pred, self.ycf_pred = self.mu_Y, self.mu_YCF
            self.loss_TCA_Y, _ = self.l2_loss(self.mu_Y, self.y, True)

        
        # regularization:
        self.loss_Reg = (1e-3 * self.wd_loss)


        # final loss
        self.loss = self.loss_TCA_Y + self.FLAGS.p_alpha * self.IPM_A + self.FLAGS.p_beta *  self.loss_CI_T + self.loss_Reg 

    def ICA_W_setting(self):

        if self.FLAGS.select_layer == 0:
            layer_num = len(self.w_I)
        else:
            layer_num = self.FLAGS.select_layer

        w_I_sum, w_C_sum, w_A_sum = self.w_I[0], self.w_C[0], self.w_A[0]
        for i in range(1, layer_num):
            w_I_sum = tf.matmul(w_I_sum, self.w_I[i])
            w_C_sum = tf.matmul(w_C_sum, self.w_C[i])
            w_A_sum = tf.matmul(w_A_sum, self.w_A[i])
        self.w_I_mean = tf.reduce_mean(tf.abs(w_I_sum), axis=1)
        self.w_C_mean = tf.reduce_mean(tf.abs(w_C_sum), axis=1)
        self.w_A_mean = tf.reduce_mean(tf.abs(w_A_sum), axis=1)

    def IPM(self):

        if self.FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        ########################################### Adjustment #########################################
        
        self.rep_A_0, self.rep_A_1 = tf.dynamic_partition(data = self.rep_A, partitions = self.partitions_i0_i1, num_partitions=2)

        mean_A_0 = tf.reduce_mean(self.rep_A_0, reduction_indices=0)
        mean_A_1 = tf.reduce_mean(self.rep_A_1, reduction_indices=0)

        self.IPM_A = tf.reduce_sum(tf.square(2.0 * p_ipm * mean_A_1 - 2.0 * (1.0 - p_ipm) * mean_A_0))

                    
    def setup_train_ops(self):
        R_vars = utils.vars_from_scopes(['representation'])
        T_vars = utils.vars_from_scopes(['treatment'])
        O_vars = utils.vars_from_scopes(['outcome'])

        T_C_vars = utils.vars_from_scopes(['treatment/treatment_C_'])

        if self.FLAGS.way_to_set_lr == 'const':
            self.train = tf.compat.v1.train.AdamOptimizer(self.lr, 0.8).minimize(self.loss, var_list=R_vars+T_vars+O_vars)
            self.train_discriminatorT_givenC = tf.compat.v1.train.AdamOptimizer(self.lr, 0.5).minimize(self.loss_C_T, var_list=T_C_vars)
        elif self.FLAGS.way_to_set_lr == 'exp_decay':
            self.train =  tf.compat.v1.train.AdamOptimizer(self.lr_exp, 0.8).minimize(self.loss, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)
            self.train_discriminatorT_givenC = tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, 0.5).minimize(self.loss_C_T, var_list=T_C_vars, global_step=self.global_step)
        elif self.FLAGS.way_to_set_lr == 'piecewise_const':
            self.train =  tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, 0.8).minimize(self.loss, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)
            self.train_discriminatorT_givenC = tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, 0.5).minimize(self.loss_C_T, var_list=T_C_vars, global_step=self.global_step)



    def representation(self, input, dim_in, dim_out, layer, name):
        rep, weight, bias = [input], [], []

        dim = np.around(np.linspace(dim_in, dim_out, layer+1)).astype(int)

        for i in range(0, layer):
            w, b = self.FC_layer(dim_in=dim[i], dim_out=dim[i+1], name='_{}_{}'.format(name, i))
            weight.append(w)
            bias.append(b)
            out = tf.add(tf.matmul(rep[i], weight[i], name='matmul_{}_{}'.format(name, i)), bias[i], name='add_{}_{}'.format(name, i))
            if self.FLAGS.batch_norm:
                out = tf.layers.batch_normalization(out, training = self.is_training)

            rep.append(tf.nn.dropout(self.activation(out), self.do_in))
        
        if self.FLAGS.rep_normalization:
            rep[-1] = rep[-1] / utils.safe_sqrt(tf.reduce_sum(tf.square(rep[-1]), axis=1, keep_dims=True))

        return rep[-1], rep, weight, bias
    
    def predict(self, input, dim_in, dim_out, layer, name, wd=0, class_num=1, mode='mu'):
        pred, weight, bias = [input], [], []

        dim = np.around(np.linspace(dim_in, dim_out, layer + 1)).astype(int)

        for i in range(0, layer):
            w, b = self.FC_layer(dim_in=dim[i], dim_out=dim[i + 1], name='_{}_{}'.format(name, i), wd=wd)
            weight.append(w)
            bias.append(b)
            out = tf.add(tf.matmul(pred[i], weight[i], name='matmul_{}_{}'.format(name, i)), bias[i], name='add_{}_{}'.format(name, i))
            pred.append(tf.nn.dropout(self.activation(out), keep_prob = self.do_out))
            

        w, b = self.FC_layer(dim_in=dim[-1], dim_out=class_num, name='_{}_{}'.format(name, 'pred'), wd=wd)
        weight.append(w)
        bias.append(b)
        out = tf.add(tf.matmul(pred[-1], weight[-1], name='matmul_{}_{}'.format(name, 'pred')), bias[-1],name='add_{}_{}'.format(name, 'pred'))
        if mode == 'mu':
            pred.append(out)
            # pred.append(tf.nn.dropout(out, self.do_out))
        else:
            pred.append(tf.nn.tanh(out))
            # pred.append(tf.nn.dropout(tf.nn.tanh(out), self.do_out))

        return pred[-1], pred, weight, bias
    
    def treatment(self, input, dim_in, dim_out, layer, name, mode='mu'):
        mu_T, mus_T, w_muT, b_muT = self.predict(input, dim_in, dim_out, layer, name, self.FLAGS.decay_rate, class_num = self.t_dim, mode = mode)
        return mu_T, mus_T, w_muT, b_muT

    def output(self, input, dim_in, dim_out, layer, name, mode='mu'):
        ''' For Binary Treatment: model Y(0) and Y(1) separately; Otherwise: Onemodel Way'''
        if self.t_dim == 2: 
            mu_Y_0, mus_Y_0, w_muY_0, b_muY_0 = self.predict(input, dim_in, dim_out, layer, name+'0', self.FLAGS.decay_rate, self.y_dim, mode)
            mu_Y_1, mus_Y_1, w_muY_1, b_muY_1 = self.predict(input, dim_in, dim_out, layer, name+'1', self.FLAGS.decay_rate, self.y_dim, mode)

            mu_YF_0, mu_YCF_0 = tf.dynamic_partition(data = mu_Y_0, partitions = self.partitions_i0_i1, num_partitions=2)
            mu_YCF_1, mu_YF_1 = tf.dynamic_partition(data = mu_Y_1, partitions = self.partitions_i0_i1, num_partitions=2)

            mu_YF = tf.dynamic_stitch([self.i0, self.i1], [mu_YF_0, mu_YF_1])
            mu_YCF = tf.dynamic_stitch([self.i0, self.i1], [mu_YCF_1, mu_YCF_0])

            mus_Y = mus_Y_0 + mus_Y_1
            w_muY = w_muY_0 + w_muY_1
            b_muY = b_muY_0 + b_muY_1

            return mu_YF, mu_YCF, mus_Y, w_muY, b_muY
        else:
            mu_Y, mus_Y, w_muY, b_muY = self.predict(input, dim_in, dim_out, layer, name, self.FLAGS.decay_rate, 1, mode)
            
            return mu_Y, mu_Y, mus_Y, w_muY, b_muY



    
    def log_loss(self, pred, label, sample = False):

        # labels = tf.concat((1 - label, label), axis=1)
        labels = label
        logits = pred


        if self.FLAGS.loss_type == 'sigmoid':
            prob = 0.995 / (1.0 + tf.exp(-pred)) + 0.0025
            ps_score = tf.reduce_sum(tf.multiply(labels, prob), axis = 1, keepdims=True)  
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        elif self.FLAGS.loss_type == 'softmax':
            prob = tf.nn.softmax(pred) 
            ps_score = tf.reduce_sum(tf.multiply(labels, prob), axis = 1, keepdims=True)  
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)              

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1)),tf.float32))

        if sample and self.FLAGS.reweight_sample:
            loss = tf.reduce_mean(self.sample_weight * loss)
        else:
            loss = tf.reduce_mean(loss)
        return prob, ps_score, loss, accuracy
    
    def l2_loss(self, pred, out, sample = False):

        if sample and self.FLAGS.reweight_sample:
            loss = tf.reduce_mean(self.sample_weight * tf.square(pred - out))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(pred - out)))
        else:
            loss = tf.reduce_mean(tf.square(pred - out))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(pred - out)))

        return loss, pred_error


    def loss_cross_entrop_two_dists_with_logits(self, logits_1, logits_2):
        if self.FLAGS.loss_type == 'sigmoid':
            prob_1 = 0.995 / (1.0 + tf.exp(-logits_1)) + 0.0025
            prob_2 = 0.995 / (1.0 + tf.exp(-logits_1)) + 0.0025
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_1, labels=prob_2)+\
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_2, labels=prob_1)
        elif self.FLAGS.loss_type == 'softmax':
            prob_1 = tf.nn.softmax(logits_1)
            prob_2 = tf.nn.softmax(logits_2)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_1, labels=prob_2)+\
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_2, labels=prob_1)
        loss = tf.reduce_mean(loss)
        return loss
    
    def record_in_tensorboard(self):

        tf.compat.v1.summary.scalar('loss/loss_TCA_Y', self.loss_TCA_Y)
        tf.compat.v1.summary.scalar('loss/IPM_A', self.IPM_A)
        tf.compat.v1.summary.scalar('loss/loss_CI_T', self.loss_CI_T)
        tf.compat.v1.summary.scalar('loss/loss_Reg', self.loss_Reg)  


















        