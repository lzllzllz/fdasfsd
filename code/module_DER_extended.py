import tensorflow as tf
import numpy as np
from functions import utils



class Net(object):
    def __init__(self, n, x_dim, y_dim, t_dim, FLAGS):
        self.n = n
        self.x_dim = x_dim
        self.t_dim = t_dim # self.t_dim = FLAGS.t_num_class
        self.y_dim = y_dim
        self.FLAGS = FLAGS

        self.wd_loss = 0

        self.x = tf.compat.v1.placeholder('float', shape=[None, x_dim], name='x')
        self.t = tf.compat.v1.placeholder('float', shape=[None, t_dim], name='t')
        self.y = tf.compat.v1.placeholder('float', shape=[None, y_dim], name='y')
        self.do_in = tf.compat.v1.placeholder("float", name='dropout_in')
        self.do_out = tf.compat.v1.placeholder("float", name='dropout_out')
        self.p_t = tf.compat.v1.placeholder("float", name='p_treated')
        self.t_threshold = tf.compat.v1.placeholder("float", name='treatment_threshold')
        self.y_threshold = tf.compat.v1.placeholder("float", name='outcome_threshold')
        self.I = tf.compat.v1.placeholder("int32", shape=[None, ], name='I')
        
        self.is_training = tf.compat.v1.placeholder(dtype = tf.bool)

        # learning rate 
        self.lr = FLAGS.lr
        self.global_step = tf.Variable(0, trainable=False)
        self.lr_exp = tf.compat.v1.train.exponential_decay(learning_rate = self.lr, global_step=self.global_step, decay_steps = FLAGS.lr_decay_steps, decay_rate = FLAGS.lr_decay_rate, staircase=True)
        self.boundaries = utils.string2list(FLAGS.lr_boundaries)
        self.values = utils.string2list(FLAGS.lr_values)
        self.lr_piecewise = tf.compat.v1.train.piecewise_constant(self.global_step, boundaries = self.boundaries, values = self.values) # 每运行一次，self.global_step+1
        
        # index partitions: i0(control) i1(treatment)
        if self.t_dim == 1:
            self.i0 = tf.cast(tf.where(self.t < self.t_threshold)[:, 0], dtype=tf.int32)
            self.i1 = tf.cast(tf.where(self.t >= self.t_threshold)[:, 0], dtype=tf.int32)
            # attention: depth = tf.shape(self.t)[0], 且不能写为self.t.shape[0]
        elif self.t_dim >= 2:
            # t[:,0] denote the encoder of Control: thus t[:,0]==1 denote Control; t[0,:]==0 denote Treatment (与上面情况正好相反)
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
            weight = tf.compat.v1.get_variable(name='weight' + name, shape=[dim_in, dim_out], initializer=self.initializer)
            if b:
                bias = tf.compat.v1.get_variable(name='bias' + name, shape=[1, dim_out], initializer=tf.constant_initializer())
        else:
            weight = tf.Variable(tf.random.normal([dim_in, dim_out], stddev=weight_init / np.sqrt(dim_in)), name='weight' + name)
            if b:
                bias = tf.Variable(tf.zeros([1, dim_out]), name='bias' + name)

        if wd>0:
            self.wd_loss += wd * tf.nn.l2_loss(weight)

        return weight, bias
    
    def build_graph(self):

        ''' Representation: A, I, C'''
        with tf.compat.v1.variable_scope('representation'):
            with tf.compat.v1.variable_scope('representation_I_'):
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
        ''' Treatment: {I->T} {A->T} {C->T} {C,I->T}'''
        with tf.compat.v1.variable_scope('treatment'):
            with tf.compat.v1.variable_scope('treatment_I_'):
                self.mu_T_I, self.mus_T_I, self.w_muT_I, self.b_muT_I = self.treatment(
                    input=self.rep_I,
                    dim_in=self.FLAGS.rep_dim,
                    dim_out=self.FLAGS.net_t_dim,
                    layer=self.FLAGS.t_layer,
                    name='Mu_Treatment_I')
            with tf.compat.v1.variable_scope('treatment_A_'):
                self.mu_T_A, self.mus_T_A, self.w_muT_A, self.b_muT_A = self.treatment(
                    input=self.rep_A,
                    dim_in=self.FLAGS.rep_dim,
                    dim_out=self.FLAGS.net_t_dim,
                    layer=self.FLAGS.t_layer,
                    name='Mu_Treatment_A')
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

        ''' Outcome: {T+C+A -> Y} {T+A -> Y} {T+C -> Y} {T+C+I -> Y}'''
        if self.t_dim == 2:
            with tf.compat.v1.variable_scope('outcome'):
                with tf.compat.v1.variable_scope('outcome_TCA_'):
                    self.mu_Y, self.mu_YCF, self.mus_Y, self.w_muY, self.b_muY = self.output(
                        input=tf.concat((self.rep_C, self.rep_A), axis=1),
                        dim_in=self.FLAGS.rep_dim * 2,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx')
                with tf.compat.v1.variable_scope('outcome_TA_'):
                    self.mu_Y_A, self.mu_YCF_A, self.mus_Y_A, self.w_muY_A, self.b_muY_A = self.output(
                        input=tf.concat(self.rep_A, axis=1),
                        dim_in=self.FLAGS.rep_dim,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx_A')   
                with tf.compat.v1.variable_scope('outcome_TC_'):
                    self.mu_Y_C, self.mu_YCF_C, self.mus_Y_C, self.w_muY_C, self.b_muY_C = self.output(
                        input=tf.concat(self.rep_C, axis=1),
                        dim_in=self.FLAGS.rep_dim,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx_C')   
                with tf.compat.v1.variable_scope('outcome_TCI_'):
                    self.mu_Y_CI, self.mu_YCF_CI, self.mus_Y_CI, self.w_muY_CI, self.b_muY_CI = self.output(
                        input=tf.concat((self.rep_C, self.rep_I), axis=1),
                        dim_in=self.FLAGS.rep_dim * 2,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx_CI')   
        else:
            with tf.compat.v1.variable_scope('outcome'):
                with tf.compat.v1.variable_scope('outcome_TCA_'):
                    self.mu_Y, self.mu_YCF, self.mus_Y, self.w_muY, self.b_muY = self.output(
                        input=tf.concat((self.rep_C, self.rep_A, self.t), axis=1),
                        dim_in=self.FLAGS.rep_dim * 2 + self.t_dim,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx')
                with tf.compat.v1.variable_scope('outcome_TA_'):
                    self.mu_Y_A, self.mu_YCF_A, self.mus_Y_A, self.w_muY_A, self.b_muY_A = self.output(
                        input=tf.concat((self.rep_A, self.t), axis=1),
                        dim_in=self.FLAGS.rep_dim + self.t_dim,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx_A')
                with tf.compat.v1.variable_scope('outcome_TC_'):
                    self.mu_Y_C, self.mu_YCF_C, self.mus_Y_C, self.w_muY_C, self.b_muY_C = self.output(
                        input=tf.concat((self.rep_A, self.t), axis=1),
                        dim_in=self.FLAGS.rep_dim + self.t_dim,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx_C')
                with tf.compat.v1.variable_scope('outcome_TCI_'):
                    self.mu_Y_CI, self.mu_YCF_CI, self.mus_Y_CI, self.w_muY_CI, self.b_muY_CI = self.output(
                        input=tf.concat((self.rep_C, self.rep_I, self.t), axis=1),
                        dim_in=self.FLAGS.rep_dim * 2 + self.t_dim,
                        dim_out=self.FLAGS.net_y_dim,
                        layer=self.FLAGS.y_layer,
                        name='Mu_ytx_CI')
        
        ''' Sample Weight: 3 ways (Learnable Params or IPW or Constant 1) '''

        
        if self.FLAGS.reweight_sample==1 and self.FLAGS.train_weights==1:
            with tf.compat.v1.variable_scope('weight'):
                sample_weight = tf.compat.v1.get_variable(name='sample_weight', shape=[self.n, 1], initializer=tf.constant_initializer(1))
                self.sample_weight_all = sample_weight 
                self.sample_weight = tf.gather(sample_weight, self.I)

        if self.FLAGS.reweight_sample==1 and self.FLAGS.train_weights==0:
            # standardize within treatment and control groups
            _, self.ps_score, _, _ = self.log_loss(self.mu_T_C, self.t)
            self.ipw_weight = 1/(self.ps_score+1E-10)
            self.ipw_weight_0, self.ipw_weight_1 = tf.dynamic_partition(data = self.ipw_weight, partitions = self.partitions_i0_i1, num_partitions=2)
            self.ipw_weight_0_std = self.ipw_weight_0/tf.reduce_sum(self.ipw_weight_0)*tf.reduce_sum(1.0 - self.t) 
            self.ipw_weight_1_std = self.ipw_weight_1/tf.reduce_sum(self.ipw_weight_1)*tf.reduce_sum(self.t)              
            self.sample_weight = tf.dynamic_stitch([self.i0, self.i1], [self.ipw_weight_0_std, self.ipw_weight_1_std])
        elif self.FLAGS.reweight_sample==0 and self.FLAGS.train_weights==0:
            self.sample_weight = tf.Variable(tf.ones(shape=[self.n, 1]))
    
    def calculate_loss(self):
        
        # calculate w_I/C/A_mean and self.IPM_I/C/A
        self.ICA_W_setting()
        self.IPM()

        ''' Components of loss ''' 
        # loss_I_T, loss_A_T
        # loss_TCA_Y, loss_TA_Y
        # loss_ICA, loss_ICA_1 

        ''' loss for predict T'''
        if self.t_dim >= 2:
            self.t_pred_I, _, self.loss_I_T, self.acc_I_T = self.log_loss(self.mu_T_I, self.t)
            self.t_pred_A, _, self.loss_A_T, self.acc_A_T = self.log_loss(self.mu_T_A, self.t)
            self.t_pred_C, _, self.loss_C_T, self.acc_C_T = self.log_loss(self.mu_T_C, self.t)
            self.t_pred_CI, _, self.loss_CI_T, self.acc_CI_T = self.log_loss(self.mu_T_CI, self.t)
        else: 
            self.t_pred_I, self.t_pred_A, self.t_pred_C, self.t_pred_CI = self.mu_T_I, self.mu_T_A, self.mu_T_C, self.mu_T_CI
            self.loss_I_T, _ = self.l2_loss(self.mu_T_I, self.t)
            self.loss_A_T, _ = self.l2_loss(self.mu_T_A, self.t)
            self.loss_C_T, _ = self.l2_loss(self.mu_T_C, self.t)
            self.loss_CI_T, _ = self.l2_loss(self.mu_T_CI, self.t)

        ''' loss for predict Y'''
        if self.y_dim >= 2:
            self.y_pred, _, self.loss_TCA_Y, _ = self.log_loss(self.mu_Y, self.y, True)
            self.y_pred_A, _, self.loss_TA_Y, _ = self.log_loss(self.mu_Y_A, self.y)
            self.y_pred_C, _, self.loss_TC_Y, _ = self.log_loss(self.mu_Y_C, self.y)
            self.y_pred_CI, _, self.loss_TCI_Y, _ = self.log_loss(self.mu_Y_CI, self.y)
            self.ycf_pred, _, _, _ = self.log_loss(self.mu_YCF, self.y)
        else:
            self.y_pred, self.y_pred_A, self.ycf_pred = self.mu_Y, self.mu_Y_A, self.mu_YCF
            self.y_pred_C, self.y_pred_CI = self.mu_Y_C, self.mu_Y_CI
            self.loss_TCA_Y, _ = self.l2_loss(self.mu_Y, self.y, True)
            self.loss_TA_Y, _ = self.l2_loss(self.mu_Y_A, self.y)
            self.loss_TC_Y, _ = self.l2_loss(self.mu_Y_C, self.y)
            self.loss_TCI_Y, _ = self.l2_loss(self.mu_Y_CI, self.y)
            

        ''' loss for net weights orthogonality'''
        self.loss_ICA = tf.reduce_sum(self.w_I_mean * self.w_C_mean) + tf.reduce_sum(self.w_I_mean * self.w_A_mean) + tf.reduce_sum(self.w_C_mean * self.w_A_mean)
        self.loss_ICA_1 = tf.square(tf.reduce_sum(self.w_I_mean) - 1.0) + tf.square(tf.reduce_sum(self.w_C_mean) - 1.0) + tf.square(tf.reduce_sum(self.w_A_mean) - 1.0)

        ''' loss to constrain p(t|A) = p(t) '''
        self.loss_A_indep = (-self.loss_A_T) * self.FLAGS.p_scale_A_indep
        
        ''' loss to constrain p(y|C, T)=p(y|I,C,T)'''
        if self.y_dim >= 2:
            self.loss_I_condindep = self.loss_cross_entrop_two_dists_with_logits(logits_1 = self.mu_Y_C, logits_2 = self.mu_Y_CI)
        else:
            self.loss_I_condindep, _ = self.l2_loss(pred = self.mu_Y_C, out = self.mu_Y_CI)
        # rescale: FLAGS.p_scale_I_condindep=1 by default
        self.loss_I_condindep = self.loss_I_condindep * self.FLAGS.p_scale_I_condindep


        ''' Combine Components '''
        # {T, C, A}--> Y
        self.loss_R = self.loss_TCA_Y

        self.loss_A = (self.loss_A_indep + self.loss_TA_Y)
        self.loss_A_ipm = (self.IPM_A + self.loss_TA_Y)

        self.loss_I = (self.loss_I_T + self.loss_I_condindep)
        self.loss_I_ipm = (self.loss_I_T + self.IPM_I)

        self.loss_O = (self.loss_ICA + self.loss_ICA_1)

        # regularization
        self.loss_Reg = (1e-3 * self.wd_loss)
        
        # main loss: 4 types
        self.loss_A_gan_I_ipm = self.loss_R + self.FLAGS.p_alpha * self.loss_A     + self.FLAGS.p_beta * self.loss_I_ipm + self.FLAGS.p_mu * self.loss_O + self.FLAGS.p_lambda * self.loss_Reg 
        self.loss_A_ipm_I_gan = self.loss_R + self.FLAGS.p_alpha * self.loss_A_ipm + self.FLAGS.p_beta * self.loss_I     + self.FLAGS.p_mu * self.loss_O + self.FLAGS.p_lambda * self.loss_Reg 
        self.loss_A_ipm_I_ipm = self.loss_R + self.FLAGS.p_alpha * self.loss_A_ipm + self.FLAGS.p_beta * self.loss_I_ipm + self.FLAGS.p_mu * self.loss_O + self.FLAGS.p_lambda * self.loss_Reg 
        self.loss_A_gan_I_gan = self.loss_R + self.FLAGS.p_alpha * self.loss_A   + self.FLAGS.p_beta * self.loss_I     + self.FLAGS.p_mu * self.loss_O + self.FLAGS.p_lambda * self.loss_Reg + self.FLAGS.p_theta * self.loss_C_T
        
        # loss to train the weights
        if self.FLAGS.reweight_sample:
            self.loss_w = tf.square(tf.reduce_sum(self.sample_weight_0)/tf.reduce_sum(1.0 - self.t) - 1.0) + tf.square(tf.reduce_sum(self.sample_weight_1)/tf.reduce_sum(self.t) - 1.0)
            self.loss_C_B = self.FLAGS.p_gamma * (self.IPM_C + self.loss_w)
            self.loss_balance = self.loss_R + self.loss_C_B + self.loss_Reg

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

        ########################################## Instrumental ########################################

        if self.t_dim == 1 and self.y_dim == 1:
            i_1_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t >= self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y >= self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
            i_1_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t >= self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y < self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
            i_0_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t < self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y >= self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
            i_0_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t < self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y < self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
        elif self.t_dim > 1 and self.y_dim > 1:
            i_1_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t[:,0] <= 0)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y[:,0] <= 0)[:, 0], dtype=tf.int32)],axis=0)))
            i_1_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t[:,0] <= 0)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y[:,0] >= 1)[:, 0], dtype=tf.int32)],axis=0)))
            i_0_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t[:,0] >= 1)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y[:,0] <= 0)[:, 0], dtype=tf.int32)],axis=0)))
            i_0_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t[:,0] >= 1)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y[:,0] >= 1)[:, 0], dtype=tf.int32)],axis=0)))
        elif self.t_dim == 1 and self.y_dim > 1:
            i_1_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t >= self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y[:,0] <= 0)[:, 0], dtype=tf.int32)],axis=0)))
            i_1_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t >= self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y[:,0] >= 1)[:, 0], dtype=tf.int32)],axis=0)))
            i_0_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t < self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y[:,0] <= 0)[:, 0], dtype=tf.int32)],axis=0)))
            i_0_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t < self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y[:,0] >= 1)[:, 0], dtype=tf.int32)],axis=0)))
        elif self.t_dim > 1 and self.y_dim == 1:
            i_1_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t[:,0] <= 0)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y >= self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
            i_1_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t[:,0] <= 0)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y < self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
            i_0_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t[:,0] >= 1)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y >= self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
            i_0_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t[:,0] >= 1)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y < self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))



        partitions_1_1 = tf.reduce_sum(tf.one_hot(i_1_1, depth = tf.shape(self.t)[0], dtype='int32'), 0)
        partitions_1_0 = tf.reduce_sum(tf.one_hot(i_1_0, depth = tf.shape(self.t)[0], dtype='int32'), 0)
        partitions_0_1 = tf.reduce_sum(tf.one_hot(i_0_1, depth = tf.shape(self.t)[0], dtype='int32'), 0)
        partitions_0_0 = tf.reduce_sum(tf.one_hot(i_0_0, depth = tf.shape(self.t)[0], dtype='int32'), 0)


        if self.FLAGS.reweight_sample==1 and self.FLAGS.train_weights==1:
            w_1_1 = tf.gather(self.sample_weight,i_1_1)
            w_1_0 = tf.gather(self.sample_weight,i_1_0)
            w_0_1 = tf.gather(self.sample_weight,i_0_1)
            w_0_0 = tf.gather(self.sample_weight,i_0_0)
        elif self.FLAGS.reweight_sample==1 and self.FLAGS.train_weights==0:
            # weights are Tensor(not Variable), using tf.gather() would raise warning
            _, w_1_1 = tf.dynamic_partition(data = self.sample_weight, partitions = partitions_1_1, num_partitions = 2)        
            _, w_1_0 = tf.dynamic_partition(data = self.sample_weight, partitions = partitions_1_0, num_partitions = 2)        
            _, w_0_1 = tf.dynamic_partition(data = self.sample_weight, partitions = partitions_0_1, num_partitions = 2)        
            _, w_0_0 = tf.dynamic_partition(data = self.sample_weight, partitions = partitions_0_0, num_partitions = 2)        
        elif self.FLAGS.reweight_sample==0 and self.FLAGS.train_weights==0:
            w_1_1 = 1
            w_1_0 = 1
            w_0_1 = 1
            w_0_0 = 1


        # Replace tf.gather With tf.dynamic_partition (tf.gather(Tensor) would raise warning)
        # I_1_1 = tf.gather(self.rep_I,i_1_1)
        # I_1_0 = tf.gather(self.rep_I,i_1_0)
        # I_0_1 = tf.gather(self.rep_I,i_0_1)
        # I_0_0 = tf.gather(self.rep_I,i_0_0)
        _, I_1_1 = tf.dynamic_partition(data = self.rep_I, partitions = partitions_1_1, num_partitions = 2)        
        _, I_1_0 = tf.dynamic_partition(data = self.rep_I, partitions = partitions_1_0, num_partitions = 2)        
        _, I_0_1 = tf.dynamic_partition(data = self.rep_I, partitions = partitions_0_1, num_partitions = 2)        
        _, I_0_0 = tf.dynamic_partition(data = self.rep_I, partitions = partitions_0_0, num_partitions = 2)
        

        mean_1_1 = tf.reduce_mean(w_1_1 * I_1_1, axis=0)
        mean_1_0 = tf.reduce_mean(w_1_0 * I_1_0, axis=0)
        mean_0_1 = tf.reduce_mean(w_0_1 * I_0_1, axis=0)
        mean_0_0 = tf.reduce_mean(w_0_0 * I_0_0, axis=0)


        mmd_1 = tf.reduce_sum(tf.square(2.0 * p_ipm * mean_1_1 - 2.0 * (1.0 - p_ipm) * mean_1_0))
        mmd_0 = tf.reduce_sum(tf.square(2.0 * p_ipm * mean_0_1 - 2.0 * (1.0 - p_ipm) * mean_0_0))

        self.IPM_I = mmd_0 + mmd_1

        ########################################### ConFounder #########################################
        
        self.rep_C_0, self.rep_C_1 = tf.dynamic_partition(data = self.rep_C, partitions = self.partitions_i0_i1, num_partitions=2)
        # self.rep_C_0 = tf.gather(self.rep_C, self.i0)
        # self.rep_C_1 = tf.gather(self.rep_C, self.i1)
 
        if self.FLAGS.reweight_sample==1 and self.FLAGS.train_weights==1:
            self.sample_weight_0 = tf.gather(self.sample_weight, self.i0)
            self.sample_weight_1 = tf.gather(self.sample_weight, self.i1)
        elif self.FLAGS.reweight_sample==1 and self.FLAGS.train_weights==0:
            self.sample_weight_0, self.sample_weight_1 = tf.dynamic_partition(data = self.sample_weight, partitions = self.partitions_i0_i1, num_partitions=2)
        else:
            self.sample_weight_0 = 1
            self.sample_weight_1 = 1

        mean_C_0 = tf.reduce_mean(self.sample_weight_0 * self.rep_C_0, reduction_indices=0)
        mean_C_1 = tf.reduce_mean(self.sample_weight_1 * self.rep_C_1, reduction_indices=0)

        self.IPM_C = tf.reduce_sum(tf.square(2.0 * p_ipm * mean_C_1 - 2.0 * (1.0 - p_ipm) * mean_C_0))

        ########################################### Adjustment #########################################
        
        self.rep_A_0, self.rep_A_1 = tf.dynamic_partition(data = self.rep_A, partitions = self.partitions_i0_i1, num_partitions=2)
        # self.rep_A_0 = tf.gather(self.rep_A, self.i0)
        # self.rep_A_1 = tf.gather(self.rep_A, self.i1)

        mean_A_0 = tf.reduce_mean(self.rep_A_0, reduction_indices=0)
        mean_A_1 = tf.reduce_mean(self.rep_A_1, reduction_indices=0)

        self.IPM_A = tf.reduce_sum(tf.square(2.0 * p_ipm * mean_A_1 - 2.0 * (1.0 - p_ipm) * mean_A_0))

                    
    def setup_train_ops(self):
        R_vars = utils.vars_from_scopes(['representation'])
        T_vars = utils.vars_from_scopes(['treatment'])
        O_vars = utils.vars_from_scopes(['outcome'])

        T_A_vars = utils.vars_from_scopes(['treatment/treatment_A_'])
        T_C_vars = utils.vars_from_scopes(['treatment/treatment_C_'])

        Y_TC_vars = utils.vars_from_scopes(['outcome/outcome_TC_'])
        Y_TCI_vars = utils.vars_from_scopes(['outcome/outcome_TCI_'])

        R_A_vars = utils.vars_from_scopes(['representation/representation_A_'])
        R_I_vars = utils.vars_from_scopes(['representation/representation_I_'])
        

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            '''' original version: self.train_balance '''
            if self.FLAGS.reweight_sample==1 and self.FLAGS.train_weights==1:
                W_vars = utils.vars_from_scopes(['weight'])
                self.train_balance = tf.compat.v1.train.AdamOptimizer(self.lr, self.FLAGS.Adam_beta_1).minimize(self.loss_balance, var_list=W_vars, global_step=self.global_step)


            '''' ancillary: train the discriminator '''
            if self.FLAGS.way_to_set_lr == 'const':
                # train the discriminator of T given A ( fix the rep A, train the predictor A->T)
                self.train_discriminatorT_givenA =  tf.compat.v1.train.AdamOptimizer(self.lr, 0.5).minimize(self.loss_A_T, var_list=T_A_vars)
                # train the discriminator of Y ( fix the rep {I, C}, train the predictor {T+C -> Y} and {T+C+I -> Y})
                self.train_discriminatorY_givenIC =  tf.compat.v1.train.AdamOptimizer(self.lr, 0.5).minimize(self.loss_TC_Y+self.loss_TCI_Y, var_list=Y_TC_vars+Y_TCI_vars)
                self.train_discriminator =  tf.compat.v1.train.AdamOptimizer(self.lr, 0.5).minimize(self.loss_A_T+self.loss_TC_Y+self.loss_TCI_Y, var_list=T_A_vars+Y_TC_vars+Y_TCI_vars)

                self.train_discriminatorT_givenC = tf.compat.v1.train.AdamOptimizer(self.lr, 0.5).minimize(self.loss_C_T, var_list=T_C_vars)
            
            elif self.FLAGS.way_to_set_lr == 'exp_decay':
                self.train_discriminatorT_givenA =  tf.compat.v1.train.AdamOptimizer(self.lr_exp, 0.5).minimize(self.loss_A_T, var_list=T_A_vars, global_step=self.global_step)
                self.train_discriminatorY_givenIC =  tf.compat.v1.train.AdamOptimizer(self.lr_exp, 0.5).minimize(self.loss_TC_Y+self.loss_TCI_Y, var_list=Y_TC_vars+Y_TCI_vars, global_step=self.global_step)
                self.train_discriminator =  tf.compat.v1.train.AdamOptimizer(self.lr_exp, 0.5).minimize(self.loss_A_T+self.loss_TC_Y+self.loss_TCI_Y, var_list=T_A_vars+Y_TC_vars+Y_TCI_vars, global_step=self.global_step)

                self.train_discriminatorT_givenC = tf.compat.v1.train.AdamOptimizer(self.lr_exp, 0.5).minimize(self.loss_C_T, var_list=T_C_vars, global_step=self.global_step)
            
            elif self.FLAGS.way_to_set_lr == 'piecewise_const':
                self.train_discriminatorT_givenA =  tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, 0.5).minimize(self.loss_A_T, var_list=T_A_vars, global_step=self.global_step)
                self.train_discriminatorY_givenIC =  tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, 0.5).minimize(self.loss_TC_Y+self.loss_TCI_Y, var_list=Y_TC_vars+Y_TCI_vars, global_step=self.global_step)
                self.train_discriminator =  tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, 0.5).minimize(self.loss_A_T+self.loss_TC_Y+self.loss_TCI_Y, var_list=T_A_vars+Y_TC_vars+Y_TCI_vars, global_step=self.global_step)
                
                self.train_discriminatorT_givenC = tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, 0.5).minimize(self.loss_C_T, var_list=T_C_vars, global_step=self.global_step)

       

            '''  main: train all the variables '''  
            if self.FLAGS.way_to_set_lr == 'const':
                self.train_A_ipm_I_ipm = tf.compat.v1.train.AdamOptimizer(self.lr, self.FLAGS.Adam_beta_1).minimize(self.loss_A_ipm_I_ipm, var_list=R_vars+T_vars+O_vars)
                self.train_A_gan_I_ipm = tf.compat.v1.train.AdamOptimizer(self.lr, self.FLAGS.Adam_beta_1).minimize(self.loss_A_gan_I_ipm, var_list=R_vars+T_vars+O_vars)
                self.train_A_gan_I_gan = tf.compat.v1.train.AdamOptimizer(self.lr, self.FLAGS.Adam_beta_1).minimize(self.loss_A_gan_I_gan, var_list=R_vars+T_vars+O_vars)
                self.train_A_ipm_I_gan = tf.compat.v1.train.AdamOptimizer(self.lr, self.FLAGS.Adam_beta_1).minimize(self.loss_A_ipm_I_gan, var_list=R_vars+T_vars+O_vars)
            elif self.FLAGS.way_to_set_lr == 'exp_decay':
                self.train_A_ipm_I_ipm = tf.compat.v1.train.AdamOptimizer(self.lr_exp, self.FLAGS.Adam_beta_1).minimize(self.loss_A_ipm_I_ipm, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)
                self.train_A_gan_I_ipm = tf.compat.v1.train.AdamOptimizer(self.lr_exp, self.FLAGS.Adam_beta_1).minimize(self.loss_A_gan_I_ipm, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)
                self.train_A_gan_I_gan = tf.compat.v1.train.AdamOptimizer(self.lr_exp, self.FLAGS.Adam_beta_1).minimize(self.loss_A_gan_I_gan, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)
                self.train_A_ipm_I_gan = tf.compat.v1.train.AdamOptimizer(self.lr_exp, self.FLAGS.Adam_beta_1).minimize(self.loss_A_ipm_I_gan, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)
            elif self.FLAGS.way_to_set_lr == 'piecewise_const':
                self.train_A_ipm_I_ipm = tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, self.FLAGS.Adam_beta_1).minimize(self.loss_A_ipm_I_ipm, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)
                self.train_A_gan_I_ipm = tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, self.FLAGS.Adam_beta_1).minimize(self.loss_A_gan_I_ipm, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)
                self.train_A_gan_I_gan = tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, self.FLAGS.Adam_beta_1).minimize(self.loss_A_gan_I_gan, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)
                self.train_A_ipm_I_gan = tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, self.FLAGS.Adam_beta_1).minimize(self.loss_A_ipm_I_gan, var_list=R_vars+T_vars+O_vars, global_step=self.global_step)


            '''  supplement: train the representation '''
            if self.FLAGS.way_to_set_lr == 'const':
                # train the representation A to be independent of T ( given the predictor A->T, train A to maximize the loss of this predictor)''' 
                self.train_repI_condindepY = tf.compat.v1.train.AdamOptimizer(self.lr, self.FLAGS.Adam_beta_1).minimize(self.loss_I_condindep, var_list=R_I_vars)
                # train the representation I s.t. p(Y|I,C,T)=p(Y|C,T)'''
                self.train_repA_indepT = tf.compat.v1.train.AdamOptimizer(self.lr, self.FLAGS.Adam_beta_1).minimize(-self.loss_A_T, var_list=R_A_vars)
            elif self.FLAGS.way_to_set_lr == 'exp_decay':
                self.train_repA_indepT = tf.compat.v1.train.AdamOptimizer(self.lr_exp, self.FLAGS.Adam_beta_1).minimize(-self.loss_A_T, var_list=R_A_vars, global_step=self.global_step)
                self.train_repI_condindepY = tf.compat.v1.train.AdamOptimizer(self.lr_exp, self.FLAGS.Adam_beta_1).minimize(self.loss_I_condindep, var_list=R_I_vars, global_step=self.global_step)
            elif self.FLAGS.way_to_set_lr == 'piecewise_const':
                self.train_repA_indepT = tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, self.FLAGS.Adam_beta_1).minimize(-self.loss_A_T, var_list=R_A_vars, global_step=self.global_step)
                self.train_repI_condindepY = tf.compat.v1.train.AdamOptimizer(self.lr_piecewise, self.FLAGS.Adam_beta_1).minimize(self.loss_I_condindep, var_list=R_I_vars, global_step=self.global_step)
                

    def representation(self, input, dim_in, dim_out, layer, name):
        rep, weight, bias = [input], [], []

        dim = np.around(np.linspace(dim_in, dim_out, layer+1)).astype(int)

        for i in range(0, layer):
            w, b = self.FC_layer(dim_in=dim[i], dim_out=dim[i+1], name='_{}_{}'.format(name, i))
            weight.append(w)
            bias.append(b)
            out = tf.add(tf.matmul(rep[i], weight[i], name='matmul_{}_{}'.format(name, i)), bias[i], name='add_{}_{}'.format(name, i))
            # if self.FLAGS.batch_norm:
            #     batch_mean, batch_var = tf.nn.moments(out, [0])
            #     out = tf.nn.batch_normalization(out, batch_mean, batch_var, 0, 1, 1e-3)
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
        # if self.FLAGS.t_is_binary:
        #     mu_T, mus_T, w_muT, b_muT = self.predict(input, dim_in, dim_out, layer, name, self.FLAGS.decay_rate, 2, mode)
        # else:
        #     mu_T, mus_T, w_muT, b_muT = self.predict(input, dim_in, dim_out, layer, name, self.FLAGS.decay_rate, 1, mode)
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
            ps_score = tf.reduce_sum(tf.multiply(labels, prob), axis = 1, keepdims=True) # 对每行求和 
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        elif self.FLAGS.loss_type == 'softmax':
            prob = tf.nn.softmax(pred) 
            ps_score = tf.reduce_sum(tf.multiply(labels, prob), axis = 1, keepdims=True) # 对每行求和 
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

        ''' cal gradient for check '''    
        # self.grad_to_Score_loss_I_condindep = tf.gradients([self.loss_I_condindep], [self.mu_Y_C, self.mu_Y_CI])
        # self.grad_to_Rep_loss_I_condindep = tf.gradients([self.loss_I_condindep], [self.rep_I, self.rep_C])
        
        # self.grad_to_Score_loss_TCI_Y = tf.gradients(self.loss_TCI_Y, self.mu_Y_CI)
        # self.grad_to_Score_loss_TC_Y = tf.gradients(self.loss_TC_Y, self.mu_Y_C)
        # self.grad_to_Rep_loss_TCI_Y = tf.gradients([self.loss_TCI_Y], [self.rep_I, self.rep_C])
        
        
        ''' tensorboard  '''
        # total loss
        tf.compat.v1.summary.scalar('total_loss_A_gan_I_ipm', self.loss_A_gan_I_ipm)
        tf.compat.v1.summary.scalar('total_loss_A_gan_I_gan', self.loss_A_gan_I_gan)
        tf.compat.v1.summary.scalar('total_loss_A_ipm_I_gan', self.loss_A_gan_I_ipm)
        tf.compat.v1.summary.scalar('total_loss_A_ipm_I_ipm', self.loss_A_ipm_I_ipm)
        tf.compat.v1.summary.scalar('total_loss/part_regression', self.loss_R)
        tf.compat.v1.summary.scalar('total_loss/part_A', self.FLAGS.p_alpha * self.loss_A)
        tf.compat.v1.summary.scalar('total_loss/part_A_ipm', self.FLAGS.p_alpha * self.loss_A_ipm)
        tf.compat.v1.summary.scalar('total_loss/part_I', self.FLAGS.p_beta * self.loss_I)
        tf.compat.v1.summary.scalar('total_loss/part_I_ipm', self.FLAGS.p_beta * self.loss_I_ipm)
        tf.compat.v1.summary.scalar('total_loss/part_O_orthogonal', self.FLAGS.p_mu * self.loss_O)
        tf.compat.v1.summary.scalar('total_loss/part_regularization', self.FLAGS.p_lambda * self.loss_Reg)
        
        

        # each components        
        ## loss/loss_TCA_Y
        tf.compat.v1.summary.scalar('loss/loss_TCA_Y', self.loss_TCA_Y)
        
        ## loss/loss_A and loss_A_ipm 
        tf.compat.v1.summary.scalar('loss/loss_A', self.loss_A)
        tf.compat.v1.summary.scalar('loss/loss_A_ipm', self.loss_A_ipm)
        ### components
        tf.compat.v1.summary.scalar('loss/loss_A/loss_TA_Y', self.loss_TA_Y)
        tf.compat.v1.summary.scalar('loss/loss_A/-loss_A_T', -self.loss_A_T)
        tf.compat.v1.summary.scalar('loss/loss_A/IPM_A', self.IPM_A)
        
        ## loss/loss_I and loss_I_ipm 
        tf.compat.v1.summary.scalar('loss/loss_I', self.loss_I)
        tf.compat.v1.summary.scalar('loss/loss_I_ipm', self.loss_I_ipm)
        ### components
        tf.compat.v1.summary.scalar('loss/loss_I/IPM_I', self.IPM_I)
        tf.compat.v1.summary.scalar('loss/loss_I/loss_I_condindep', self.loss_I_condindep)
        tf.compat.v1.summary.scalar('loss/loss_I/loss_I_T', self.loss_I_T)

        ## loss_O (orthogonality)
        tf.compat.v1.summary.scalar('loss/loss_O', self.loss_O)
        ### components
        tf.compat.v1.summary.scalar('loss/loss_O/loss_ICA', self.loss_ICA)
        tf.compat.v1.summary.scalar('loss/loss_O/loss_ICA_1', self.loss_ICA_1)

        ## components of loss_Reg (regularization)
        tf.compat.v1.summary.scalar('loss/loss_Reg', self.loss_Reg)

                    
        ## predictive loss
        tf.compat.v1.summary.scalar('pred Y/loss_Y_TA', self.loss_TA_Y)
        tf.compat.v1.summary.scalar('pred Y/loss_Y_TC', self.loss_TC_Y)
        tf.compat.v1.summary.scalar('pred Y/loss_Y_TCA', self.loss_TCA_Y)
        tf.compat.v1.summary.scalar('pred Y/loss_Y_TCI', self.loss_TCI_Y)
        
        tf.compat.v1.summary.scalar('pred T/loss_T_A', self.loss_A_T)
        tf.compat.v1.summary.scalar('pred T/loss_T_I', self.loss_I_T)

        ## acc
        if self.t_dim >= 2:
            tf.compat.v1.summary.scalar('acc_A_T', self.acc_A_T)
            tf.compat.v1.summary.scalar('acc_I_T', self.acc_I_T)

        ## gradient
        # tf.compat.v1.summary.histogram('gradient/grad_of_mu_Y_C/loss_I_condindep', self.grad_to_Score_loss_I_condindep[0])
        # tf.compat.v1.summary.histogram('gradient/grad_of_mu_Y_CI/loss_I_condindep', self.grad_to_Score_loss_I_condindep[1])
        # tf.compat.v1.summary.histogram('gradient/grad_of_mu_Y_C/loss_TC_Y', self.grad_to_Score_loss_TC_Y[0])
        # tf.compat.v1.summary.histogram('gradient/grad_of_mu_Y_CI/loss_TCI_Y', self.grad_to_Score_loss_TCI_Y[0])


