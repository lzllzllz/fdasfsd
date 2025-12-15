import tensorflow as tf, pandas as pd, numpy as np, matplotlib.pyplot as plt, random, os, csv, copy
from functions import utils
from functions.utils import plot_weight, plot_uplift, get_str_from_FLAGS, get_accuracy, get_mse, get_auc

from module_DER_extended import Net
from module_DR import Net as Net_DR
from sklearn.metrics import classification_report  ##导入分类报告函数
from sklift.metrics import qini_auc_score

# hyperparameter (parameter_name, default_value, parameter_description)
FLAGS = tf.compat.v1.app.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.compat.v1.app.flags.DEFINE_float('p_alpha', 1e-1, "loss-A")
tf.compat.v1.app.flags.DEFINE_float('p_beta', 1, "loss-I")
tf.compat.v1.app.flags.DEFINE_float('p_gamma', 1, "loss-C_B")
tf.compat.v1.app.flags.DEFINE_float('p_mu', 10, "loss-O")
tf.compat.v1.app.flags.DEFINE_float('p_lambda', 1, "loss-Reg")
tf.compat.v1.app.flags.DEFINE_float('p_theta', 1, "loss_C_T")##这个多的

tf.compat.v1.app.flags.DEFINE_float('lr', 0.001, "Learning rate")
tf.compat.v1.app.flags.DEFINE_float('decay_rate', 1.0, "Weight decay rate")
tf.compat.v1.app.flags.DEFINE_integer('seed', 888, "seed")
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 256, "batch_size")
tf.compat.v1.app.flags.DEFINE_integer('num_experiments', 1, "num_experiments")
tf.compat.v1.app.flags.DEFINE_integer('rep_dim', 256, "The dimension of representation network")
tf.compat.v1.app.flags.DEFINE_integer('rep_layer', 2, "The number of representation network layers")

tf.compat.v1.app.flags.DEFINE_integer('net_t_dim', 128, "The dimension of treatment network")
tf.compat.v1.app.flags.DEFINE_integer('t_layer', 5, "The number of treatment network layers")
tf.compat.v1.app.flags.DEFINE_integer('net_y_dim', 128, "The dimension of outcome network")
tf.compat.v1.app.flags.DEFINE_integer('y_layer', 5, "The number of outcome network layers")
tf.compat.v1.app.flags.DEFINE_integer('select_layer', 0, "contribution layer")
tf.compat.v1.app.flags.DEFINE_string('activation', 'elu', "Activation function")

tf.compat.v1.app.flags.DEFINE_string('data_path', './data/Syn_16_16_16_3000/data_discrete.npz', "data")###数据
tf.compat.v1.app.flags.DEFINE_string('output_dir', './results/simulation', "output")
tf.compat.v1.app.flags.DEFINE_string('var_from', 'get_variable', "get_variable/Variable")
tf.compat.v1.app.flags.DEFINE_integer('reweight_sample', 0, "sample balance")
tf.compat.v1.app.flags.DEFINE_integer('use_p_correction', 1, "fix coef")
tf.compat.v1.app.flags.DEFINE_integer('batch_norm', 0, "batch normalization")
tf.compat.v1.app.flags.DEFINE_integer('rep_normalization', 0, "representation normalization")
tf.compat.v1.app.flags.DEFINE_integer('train_steps', 3000, "the number of steps in training in one experiment")
# tf.compat.v1.app.flags.DEFINE_integer('train_epochs', 30, "the number of epochs in training the experiment")

'''parameters for DER_extended'''
tf.compat.v1.app.flags.DEFINE_string('loss_type', 'sigmoid', "name of loss for discrete variables: sigmoid or softmax")
tf.compat.v1.app.flags.DEFINE_float('p_scale_I_condindep', 1, "scale loss_I_condindep")#I的条件独立损失的缩放系数
tf.compat.v1.app.flags.DEFINE_float('p_scale_A_indep', 1, "scale loss_A_indep")#A的独立损失的缩放系数
tf.compat.v1.app.flags.DEFINE_float('Adam_beta_1', '0.5', "parameters in tf.AdamOptimizer")

tf.compat.v1.app.flags.DEFINE_integer('steps_discriminate', 3, "the number of steps of training discriminator")
tf.compat.v1.app.flags.DEFINE_integer('train_A_separately', 1, "whether train A in 补充任务 separately to maximize loss_A_T")
tf.compat.v1.app.flags.DEFINE_integer('train_I_separately', 0, "whether train I in 补充任务 separately to minimize loss_I_condindep")
tf.compat.v1.app.flags.DEFINE_string('way_to_constrain_A', 'GAN', "contrain A to be independent of T by IPM or GAN")
tf.compat.v1.app.flags.DEFINE_string('way_to_constrain_I', 'GAN', "contrain I to be conditionally independent of Y (given C and T) by IPM or GAN")

# if reweight_sample=1 and train_weights=1, then set sample_weights as parameters to learn
# if reweight_sample=1 and train_weights=0, then set sample_weights = 1/ps(X_c) 
# if reweight_sample=0 and train_weights=0, then set sample_weights = 1 (恒等于1)
tf.compat.v1.app.flags.DEFINE_integer('train_weights', 0, "whether take the unit-weights as parameters to learn")
tf.compat.v1.app.flags.DEFINE_string('way_to_set_lr', 'piecewise_const', "const or exp_decay or piecewise_const")
tf.compat.v1.app.flags.DEFINE_integer('lr_decay_steps', '100', "parameters in tf.train.exponential_decay")
tf.compat.v1.app.flags.DEFINE_float('lr_decay_rate', '1', "parameters in tf.train.exponential_decay")
tf.compat.v1.app.flags.DEFINE_string('lr_boundaries', '[8000]', "parameters in tf.train.piecewise_constant") # global_step = i的倍数 因为每一个step包括若干个trainer
tf.compat.v1.app.flags.DEFINE_string('lr_values', '[0.001, 0.0008]', "parameters in tf.train.piecewise_constant")

tf.compat.v1.app.flags.DEFINE_string('config_name', 'config_3', "name of config, connect the results with the detailed configurations")
tf.compat.v1.app.flags.DEFINE_string('model_name', 'DER', "name of model, allowing ")

# 父类：使用decomposed representation learning估计ITE的一类模型
class RepModel_Decomposed():
    '''
    父类的构造函数为空 每个子类(具体模型)写自己的构造函数
    父类的 validate_representation_learning, pred, save_rep 这三个函数在不同的子类中是共享的
    '''
    def __init__(self):
        pass 

        
    def validate_representation_learning(self, plot=True, save_to_csv=False, indices_all = range(0,48),\
        indices_x_for_I=range(0,16), indices_x_for_C=range(16,32), indices_x_for_A=range(32,48), num_experiment=0):
        '''
        使用 w_I, w_C, w_A 验证学习到的表征是否和真实的数据生成过程相对应
        例如:w_I = net.w_I_mean (是I(X)的【每一层的网络权重系数矩阵相乘之后的绝对值】在I(X)维度求平均得到的向量的平均值
            比如说 I(X) is m-dimensional, X is p-dimensional, 那么每一层的权重系数矩阵相乘之后是 p*m 维的矩阵
            取绝对值后在行方向上取均值，则得到 p*1 的向量，反映了每一个变量对于I(X)的贡献程度
        
        该方法函数 对I, C, A之中的每一类表征所对应的 w_I, w_C, w_A 计算正确维度上的求和以及错误维度上的求和
        例如:I(X)是X的前16维的函数，那么正确维度是前16维度，错误维度是其他
        '''
        weight = pd.DataFrame({'w_I': self.w_I, 'w_C': self.w_C, 'w_A': self.w_A})
        ###保存csv文件
        weight.to_csv(os.path.join(FLAGS.output_dir, 'model_validate', 'Rep_w_{}_{}_exp_{}.csv'.format(self.model_name, FLAGS.config_name, num_experiment)), index=False)
        ##如果需要绘图，则调用 plot_weight 函数绘制权重图。
        if plot:
            plot_weight(weight, output_path=os.path.join(FLAGS.output_dir, 'model_validate', 'Validate_Rep_w_{}_{}_exp_{}.png'.format(self.model_name, FLAGS.config_name, num_experiment)))
        # 计算各个表征的网络权重在 正确映射区域 和 错误映射区域 的均值差
        len_all = len(indices_all)
        w_I_true = sum([self.w_I[i] for i in indices_x_for_I])/len(indices_x_for_I)
        w_I_false = sum([self.w_I[i] for i in set(indices_all) - set(indices_x_for_I)])/(len_all - len(indices_x_for_I))

        w_C_true = sum([self.w_C[i] for i in indices_x_for_C])/len(indices_x_for_C)
        w_C_false = sum([self.w_C[i] for i in set(indices_all) - set(indices_x_for_C)])/(len_all - len(indices_x_for_C))

        w_A_true = sum([self.w_A[i] for i in indices_x_for_A])/len(indices_x_for_A)
        w_A_false = sum([self.w_A[i] for i in set(indices_all) - set(indices_x_for_A)])/(len_all - len(indices_x_for_A))

        ###保存到CSV
        if save_to_csv:
            w_true_false = pd.DataFrame({
                'w_I_true': [w_I_true],
                'w_I_false': [w_I_false],
                'w_C_true': [w_C_true],
                'w_C_false': [w_C_false],
                'w_A_true': [w_A_true],
                'w_C_false': [w_A_false]
            })
            w_true_false.to_csv(os.path.join(FLAGS.output_dir, 'model_validate', 'Rep_w_summary_{}_{}_exp_{}.csv'.format(self.model_name, FLAGS.config_name)), index=False)

        return w_I_true, w_I_false, w_C_true, w_C_false, w_A_true, w_A_false       

    def pred(self, test, target = 'y', verbose=False):
        '''
        如果target是数值型，返回预测值
        如果target是离散型，返回每一个类的预测概率值
        '''
        net = self.net
        log = self.log

        test_dict = {net.x: test['x'], net.t: test['t'], net.y: test['y'],
            net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: test['I'],
            net.t_threshold: self.t_threshold, net.y_threshold: self.y_threshold, net.is_training: False}

        if target=='y':
            y_hat = self.sess.run(net.y_pred, feed_dict=test_dict)
            if verbose:
                log.log('=== predictive perfromance of Y:')
                log.log(classification_report(test['y'], y_hat>0.5))
            return y_hat
        elif target=='treatment':
            t_hat = self.sess.run(net.t_pred_C, feed_dict=test_dict)
            if verbose:
                log.log('=== predictive perfromance of t:')
                log.log(classification_report(test['t'], t_hat>0.5))
            return t_hat

    def save_rep(self, test, data_name='valid_data', num_experiment=0):

        model_name=self.model_name
        net = self.net
        log = self.log

        # test_dict
        test_dict = {net.x: test['x'], net.t: test['t'], net.y: test['y'],
                    net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: test['I'],
                    net.t_threshold: self.t_threshold, net.y_threshold: self.y_threshold, net.is_training: False}
        # rep_I, rep_C, rep_A, weight, weight_ipw = self.sess.run([net.rep_I,  net.rep_A, net.rep_C, net.sample_weight, net.sample_weight_ipw], feed_dict=test_dict)
        rep_I, rep_C, rep_A, weight = self.sess.run([net.rep_I,  net.rep_A, net.rep_C, net.sample_weight], feed_dict=test_dict)
        log.log('shape of representation I:'+str(rep_I.shape))
        log.log('shape of representation C:'+str(rep_C.shape))
        log.log('shape of representation A:'+str(rep_A.shape))
        
        # save results
        file_name = 'Rep_{}_{}_{}_{}_exp_{}.csv'
        file_path = os.path.join(FLAGS.output_dir, 'model_estimates')
        log.log('save results in ' + file_path)
        np.savetxt(os.path.join(file_path, file_name.format('I', data_name, model_name, FLAGS.config_name, num_experiment)), rep_I, delimiter=',')
        np.savetxt(os.path.join(file_path, file_name.format('C', data_name, model_name, FLAGS.config_name, num_experiment)), rep_C, delimiter=',')
        np.savetxt(os.path.join(file_path, file_name.format('A', data_name, model_name, FLAGS.config_name, num_experiment)), rep_A, delimiter=',')
        np.savetxt(os.path.join(file_path, 'Weight_{}_{}_{}_exp_{}.csv'.format(data_name, model_name, FLAGS.config_name, num_experiment)), weight, delimiter=',', header='weight')
        # np.savetxt(os.path.join(file_path, 'Weight_IPW_{}_{}_{}.csv'.format(data_name, model_name, FLAGS.config_name)), weight_ipw, delimiter=',', header='weight')



class RepModel_DER(RepModel_Decomposed):

    def __init__(self, train, valid, log, model_name = 'DER', num_experiment=0):

        log.log('\n== Model Step: {}'.format(model_name))
        self.log = log
        self.model_name = model_name

        ''' Set random seed '''
        random.seed(FLAGS.seed)
        tf.compat.v1.set_random_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)  

        ''' Train '''
        self.net = self.train_net(train, valid, num_experiment=num_experiment)

    def train_net(self, train, valid, num_experiment=0):
        log = self.log

        ''' Set random seed '''
        log.log("Set random seed: {}".format(FLAGS.seed))
        random.seed(FLAGS.seed)
        tf.compat.v1.set_random_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        ''' Session '''
        log.log("Session: Open ")
        tf.compat.v1.reset_default_graph()
        graph = tf.compat.v1.get_default_graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(graph=graph, config=config)
        log.log("Session: Create")

        ''' load data '''
        log.log("load data")
        D = [train['x'], train['t'], train['y'], train['I']]
        G = utils.batch_G(D, batch_size=FLAGS.batch_size)
        self.n, self.x_dim, self.y_dim, self.t_dim =  train['x'].shape[0],train['x'].shape[1], train['y'].shape[1], train['t'].shape[1]
        # print('t_dim={} y_dim={}'.format(self.t_dim, self.y_dim))
        log.log("Initialize all parameters")
        
        net = Net(self.n, self.x_dim, self.y_dim, self.t_dim, FLAGS)


        ''' Merge all the summaries '''
        merged = tf.compat.v1.summary.merge_all()
        name_tensorboard = os.path.join(FLAGS.output_dir,  'tensorboard', self.model_name+'_'+FLAGS.config_name+'_'+'exp_'+str(num_experiment)+utils.get_timestamp(precision='day'))
        # name_tensorboard = os.path.join(FLAGS.output_dir,  'tensorboard', self.model_name+'_temp')
        train_writer = tf.compat.v1.summary.FileWriter(name_tensorboard, sess.graph)
        # print('tensorboard write into ', name_tensorboard)

        ''' Initialize all parameters '''
        tf.compat.v1.global_variables_initializer().run(session=sess)
        log.log("tf: Complete")


        ''' feed_dict '''
        if self.t_dim==1:
            self.t_threshold = np.median(train['t'])
        else:
            self.t_threshold = 0.5
        if self.y_dim==1:
            self.y_threshold = np.median(train['y'])
        else:
            self.y_threshold= 0.5
        
        valid_dict = {net.x: valid['x'], net.t: valid['t'], net.y: valid['y'],
                    net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: valid['I'], net.is_training: False,
                    net.t_threshold: self.t_threshold, net.y_threshold: self.y_threshold}
                

        log.log("training ---- ")
        train_steps = FLAGS.train_steps
        # train_epochs = FLAGS.train_epochs
        # G.batch.__next__目前遵循的规则是如果长度超过len, 则batch_count重置为0, 以及shuffle data & 从头再开始取
  

        # for e in range(train_epochs):
        #     for i in range(runs_per_epoch):
        for i in range(train_steps):

            x, t, y, I = G.batch.__next__()

            batch_dict = {net.x: x, net.t: t, net.y: y, net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: I, \
                          net.t_threshold: self.t_threshold, net.y_threshold: self.y_threshold, net.is_training: True}

            # 训练过程

            # 1. 辅助任务
            # 固定representation A，训练 A-->T 的分类器
            # 固定representation C+I, 训练 {T+C->Y} 和 {T+C+I->Y} 的分类器

            if FLAGS.way_to_constrain_A=='GAN' and FLAGS.way_to_constrain_I=='IPM':
                if i==0: log.log('== 辅助任务: train_discriminatorT_givenA')
                for _ in range(FLAGS.steps_discriminate):
                    sess.run(net.train_discriminatorT_givenA, feed_dict=batch_dict)
            elif FLAGS.way_to_constrain_I=='GAN' and FLAGS.way_to_constrain_A=='IPM':
                if i==0 : log.log('== 辅助任务: train_discriminatorY_givenIC')
                for _ in range(FLAGS.steps_discriminate):
                    sess.run(net.train_discriminatorY_givenIC, feed_dict=batch_dict)
            elif FLAGS.way_to_constrain_I=='GAN' and FLAGS.way_to_constrain_A=='GAN':
                if i==0 : log.log('== 辅助任务: train_discriminator_both')
                for _ in range(FLAGS.steps_discriminate):
                    sess.run(net.train_discriminator, feed_dict=batch_dict)
            
            if FLAGS.reweight_sample==1:
                sess.run(net.train_discriminatorT_givenC, feed_dict=batch_dict)

            # 2. 主线任务: loss_A_ipm/gan_I_ipm/gan (4 types)
            # loss = loss_R + loss_A(_ipm) + loss_I(_ipm) + loss_O + loss_Reg
            #       loss_A = {loss_TA_Y + (-loss_A_T)}*p_alpha
            #       loss_A_ipm = {loss_TA_Y + IPM_A}*p_alpha
            #       loss_I = {loss_I_Y + loss_I_condindep}*p_alpha 
            #       loss_I_ipm = {loss_I_Y + IPM_I}*p_alpha

            if FLAGS.way_to_constrain_A=='GAN' and FLAGS.way_to_constrain_I=='IPM':
                if i==0: log.log('== 主线任务: train_A_gan_I_ipm')
                sess.run(net.train_A_gan_I_ipm, feed_dict=batch_dict)
            elif FLAGS.way_to_constrain_A=='GAN' and FLAGS.way_to_constrain_I=='GAN':
                if i==0: log.log('== 主线任务: train_A_gan_I_gan')
                sess.run(net.train_A_gan_I_gan, feed_dict=batch_dict)
            elif FLAGS.way_to_constrain_A=='IPM' and FLAGS.way_to_constrain_I=='GAN':
                if i==0: log.log('== 主线任务: train_A_ipm_I_gan')
                sess.run(net.train_A_ipm_I_gan, feed_dict=batch_dict)
            elif FLAGS.way_to_constrain_A=='IPM' and FLAGS.way_to_constrain_I=='IPM':
                if i==0: log.log('== 主线任务: train_A_ipm_I_ipm')
                sess.run(net.train_A_ipm_I_ipm, feed_dict=batch_dict)


            # 3. 补充任务
            # 固定A-->T, 训练 {A}, maximize loss_A_T
            # 固定 {T+C->Y} 和 {T+C+I->Y}, 训练 {I}, minimize dist{p(y|t,c), p(y|t,c,i)}
            if FLAGS.train_A_separately==1 and FLAGS.way_to_constrain_A=='GAN':
                if i==0: log.log('== 补充任务: train_repA_indepT')
                sess.run(net.train_repA_indepT, feed_dict=batch_dict)
            if FLAGS.train_I_separately==1 and FLAGS.way_to_constrain_I=='GAN':
                if i==0: log.log('== 补充任务: train_repI_condindepY')
                sess.run(net.train_repI_condindepY, feed_dict=batch_dict)
            if FLAGS.reweight_sample==1 and FLAGS.train_weights==1:
                if i==0: log.log('== 补充任务: train_weights')
                sess.run(net.train_balance, feed_dict=batch_dict)

            # 中间保存至tensorboard的结果
            if i % 10 == 0:
                summary = sess.run(merged, feed_dict=batch_dict)
                # e从0开始，不需要e-1
                train_writer.add_summary(summary, i)

            # 中间输出结果
            if i % 100 == 0:
                y_hat, t_hat = sess.run([net.y_pred, net.t_pred_I], feed_dict=batch_dict)
                valid_y_hat, valid_t_hat = sess.run([net.y_pred, net.t_pred_I], feed_dict=valid_dict)
                global_step, lr_piecewise, lr_exp = sess.run([net.global_step, net.lr_piecewise, net.lr_exp])
                loss_TCA_Y, loss_C_T, loss_A, loss_TA_Y, loss_A_indep, loss_I, loss_I_T, loss_I_condindep = \
                    sess.run([net.loss_TCA_Y, net.loss_C_T,net.loss_A, net.loss_TA_Y, net.loss_A_indep, net.loss_I, net.loss_I_T, net.loss_I_condindep], feed_dict = batch_dict)
      
                if self.y_dim >= 2:
                    y_acc_train, y_acc_valid = get_accuracy(y, y_hat), get_accuracy(valid['y'], valid_y_hat) 
                elif self.y_dim == 1:
                    y_acc_train, y_acc_valid = get_mse(y, y_hat), get_mse(valid['y'], valid_y_hat) 

                if self.t_dim >= 2:
                    t_acc_train, t_acc_valid = get_accuracy(t, t_hat), get_accuracy(valid['t'], valid_t_hat)
                elif self.y_dim ==1:
                    t_acc_train, t_acc_valid = get_mse(y, y_hat), get_mse(valid['y'], valid_y_hat) 
                
                # recod in logs
                log.log('=== i={} global_step={} lr={} lr_piecewise={} lr_exp={}'.format(i, global_step, FLAGS.lr, lr_piecewise, lr_exp))
                log.log("    y_acc on train: {}, t_acc on train: {} \n    y_acc on valid: {}, t_acc on valid: {}".format(y_acc_train, t_acc_train, y_acc_valid, t_acc_valid))
                log.log("    loss_TCA_Y={} loss_C_T={}\n    loss_A={}, loss_TA_Y={}, loss_A_indep={}\n    loss_I={}, loss_I_T={}, loss_I_condindep={}".format(loss_TCA_Y, loss_C_T, loss_A, loss_TA_Y, loss_A_indep, loss_I, loss_I_T, loss_I_condindep))

        self.w_I, self.w_C, self.w_A = sess.run([net.w_I_mean, net.w_C_mean, net.w_A_mean], feed_dict={net.do_in: 1.0})
        self.sess = sess
        
        return net  

class RepModel_DR(RepModel_Decomposed):

    def __init__(self, train, valid, log, model_name = 'DR', num_experiment=0):

        log.log('\n== Model Step: {}'.format(model_name))
        self.log = log
        self.model_name = model_name

        ''' Set random seed '''
        random.seed(FLAGS.seed)
        tf.compat.v1.set_random_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)  

        ''' Train '''
        self.net = self.train_net(train, valid, num_experiment=num_experiment)

    def train_net(self, train, valid, num_experiment=0):
        log = self.log

        ''' Set random seed '''
        log.log("Set random seed: {}".format(FLAGS.seed))
        random.seed(FLAGS.seed)
        tf.compat.v1.set_random_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        ''' Session '''
        log.log("Session: Open ")
        tf.compat.v1.reset_default_graph()
        graph = tf.compat.v1.get_default_graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(graph=graph, config=config)
        log.log("Session: Create")

        ''' load data '''
        log.log("load data")
        D = [train['x'], train['t'], train['y'], train['I']]
        G = utils.batch_G(D, batch_size=FLAGS.batch_size)
        self.n, self.x_dim, self.y_dim, self.t_dim =  train['x'].shape[0],train['x'].shape[1], train['y'].shape[1], train['t'].shape[1]
        # print('t_dim={} y_dim={}'.format(self.t_dim, self.y_dim))
        log.log("Initialize all parameters")
        
        net = Net_DR(self.n, self.x_dim, self.y_dim, self.t_dim, FLAGS)


        ''' Merge all the summaries '''
        merged = tf.compat.v1.summary.merge_all()
        name_tensorboard = os.path.join(FLAGS.output_dir,  'tensorboard', self.model_name+'_'+FLAGS.config_name+'_'+'exp_'+str(num_experiment)+utils.get_timestamp(precision='day'))
        # name_tensorboard = os.path.join(FLAGS.output_dir,  'tensorboard', self.model_name+'_temp')
        train_writer = tf.compat.v1.summary.FileWriter(name_tensorboard, sess.graph)
        # print('tensorboard write into ', name_tensorboard)

        ''' Initialize all parameters '''
        tf.compat.v1.global_variables_initializer().run(session=sess)
        log.log("tf: Complete")


        ''' feed_dict '''
        if self.t_dim==1:
            self.t_threshold = np.median(train['t'])
        else:
            self.t_threshold = 0.5
        if self.y_dim==1:
            self.y_threshold = np.median(train['y'])
        else:
            self.y_threshold= 0.5
        
        valid_dict = {net.x: valid['x'], net.t: valid['t'], net.y: valid['y'],
                    net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: valid['I'], net.is_training: False,
                    net.t_threshold: self.t_threshold, net.y_threshold: self.y_threshold}
                

        log.log("training ---- ")
        train_steps = FLAGS.train_steps
        # train_epochs = FLAGS.train_epochs
        # G.batch.__next__目前遵循的规则是如果长度超过len, 则batch_count重置为0, 以及shuffle data & 从头再开始取


        for i in range(train_steps):

            x, t, y, I = G.batch.__next__()

            batch_dict = {net.x: x, net.t: t, net.y: y, net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: I, \
                          net.t_threshold: self.t_threshold, net.y_threshold: self.y_threshold, net.is_training: True}

            # 训练过程
            if i==0: 
                log.log('== 训练: net.train 以及 net.train_discriminatorT_givenC') 

            sess.run(net.train, feed_dict=batch_dict)
            sess.run(net.train_discriminatorT_givenC, feed_dict=batch_dict)

            # 中间保存至tensorboard的结果
            if i % 10 == 0:
                summary = sess.run(merged, feed_dict=batch_dict)
                # e从0开始，不需要e-1
                train_writer.add_summary(summary, i)

            # 中间输出结果
            if i % 100 == 0:
                y_hat, t_hat = sess.run([net.y_pred, net.t_pred_CI], feed_dict=batch_dict)
                valid_y_hat, valid_t_hat = sess.run([net.y_pred, net.t_pred_CI], feed_dict=valid_dict)
                global_step, lr_piecewise, lr_exp = sess.run([net.global_step, net.lr_piecewise, net.lr_exp])
                # self.loss = self.loss_TCA_Y + self.FLAGS.p_alpha * self.IPM_A + self.FLAGS.p_beta *  self.loss_CI_T + self.loss_Reg 

                loss, loss_C_T, loss_CI_T, loss_TCA_Y, loss_IPM_A= \
                    sess.run([net.loss, net.loss_C_T,net.loss_CI_T, net.loss_TCA_Y, net.IPM_A], feed_dict = batch_dict)
      
                if self.y_dim >= 2:
                    y_acc_train, y_acc_valid = get_accuracy(y, y_hat), get_accuracy(valid['y'], valid_y_hat) 
                elif self.y_dim == 1:
                    y_acc_train, y_acc_valid = get_mse(y, y_hat), get_mse(valid['y'], valid_y_hat) 

                if self.t_dim >= 2:
                    t_acc_train, t_acc_valid = get_accuracy(t, t_hat), get_accuracy(valid['t'], valid_t_hat)
                elif self.y_dim ==1:
                    t_acc_train, t_acc_valid = get_mse(y, y_hat), get_mse(valid['y'], valid_y_hat) 
                
                # recod in logs
                log.log('=== i={} global_step={} lr={} lr_piecewise={} lr_exp={}'.format(i, global_step, FLAGS.lr, lr_piecewise, lr_exp))
                log.log("    y_acc on train: {}, t_acc on train: {} \n    y_acc on valid: {}, t_acc on valid: {}".format(y_acc_train, t_acc_train, y_acc_valid, t_acc_valid))
                log.log("    loss={}, loss_C_T={}, loss_CI_T={}, loss_TCA_Y={}, loss_IPM_A={}".format(loss, loss_C_T, loss_CI_T, loss_TCA_Y, loss_IPM_A))

        self.w_I, self.w_C, self.w_A = sess.run([net.w_I_mean, net.w_C_mean, net.w_A_mean], feed_dict={net.do_in: 1.0})
        self.sess = sess
        
        return net  

    
# 对Binary treatment， 按照 转化率 进行评估
class Evaluator():
    '''
    evaluation for categorical Y and T by conversion
    Y & T:  
        both by one-hot-encoder
    self.evaluate_by_conversion(model, data_eval, model_name):
        do not distinguish between different Y in evaluation
        只区分Y的class 0 和其他类别 ( Y的第一列为1表示不转化, 其他情况均视为转化 )

    '''

    def __init__(self, log):
        self.log=log

    def evaluate_by_factual_response(self, model, data_eval, model_name = 'DER', save_results_to_csv=False, num_experiment=0):
        """
        通过观测样本上Y预测效果评判模型
            如果Y是连续的，则返回MSE
            如果Y是离散的且二值，则ACC和AUC
            如果Y是离散的且多类(>2类), 则返回AUC
        返回结果的类型是dictionary，其中字典的key为指标的名称，字典的value为指标的计算值
        """
        self.log.log('\n== Evaluation step \n-- evaluate by factual response')

        self.y_dim = data_eval['y'].shape[1]

        # y_hat是模型的预测概率
        y_hat = model.pred(data_eval, target = 'y') 

        if self.y_dim == 1:
            mse = get_mse(data_eval['y'], y_hat)
            factual_result = pd.DataFrame({'mse': [mse]})
            res = {'mse': mse}
        else:
            acc = get_accuracy(data_eval['y'], y_hat)
            if self.y_dim ==2:
                auc = get_auc(y_hat, data_eval['y'])
                factual_result = pd.DataFrame({'acc': [acc], 'auc': [auc]})
                res = {'acc': acc, 'auc': auc}
            else:
                factual_result = pd.DataFrame({'acc': [acc]}) 
                res = {'acc': acc} 

        # 保留单次的结果至csv
        if save_results_to_csv:
            factual_result.to_csv(os.path.join(FLAGS.output_dir,'model_evaluate','Evaluate_summary_by_factual_response_{}_{}_exp_{}.csv'.format(model_name, FLAGS.config_name, num_experiment)), index=None)
        
        return res

    def evaluate_by_uplift(self, model, data_eval, model_name = 'DeR', num_experiment=0, save_detailed_results_to_csv=False, save_results_to_csv=False):
        '''
        Attention:  当前计算方式适用于 T 和 Y 均为离散分类变量
                    且Y编码类似于 的第1列=不分期 其余列=各类分期类型 (因为计算转化率时=1-前者)
        
        通过观测样本上增益值预测效果的排序性评判模型
        ## 当前代码仅适用于 T 和 Y 均为离散分类变量的情况，未来需要扩展到Y为连续且有单调性的情况 (Y取值越大越好 or Y取值越小越好)
        ## 且当前计算转化率时, 认为 Y 的index=0的对应不转化，其他列对应转化 (index0=是否不分期,其余列=各类分期类型,计算转化率时=1-不分期概率)
            
            如果T为离散二值，则返回 qini_score, conv_rate_model(模型策略下的转化率), covn_rate_random(数据本身的转化率, 评估数据要求策略为随机发放)
            如果T是离散的且多类(>2类), 则返回 qini_score_list(每一种treatment都要和对照组进行对比，计算qini_score) , conv_rate_model(模型策略下的转化率), covn_rate_random(数据本身的转化率, 评估数据要求策略为随机发放)

            同时该方法函数还调用utils.plot_uplit画qini curve图和uplift条形图
        
        返回结果的类型是dictionary, 其中字典的key为指标的名称, 字典的value为指标的计算值. (当计算值为qini_score_list时, 先转化成`;`连接的字符串)
        '''

        self.log.log('\n== Evaluation step \n-- evaluate by uplift prediction')

        # Calculate Conversion Under Each Treatment (1-P(Y=0))
        self.log.log('\n=== 计算每个样本点在 各个treatment下 的增益 (转化率) :')

        # 取 n样本量, t的维度, y的维度 
        self.n, self.t_dim, self.y_dim = data_eval['t'].shape[0], data_eval['t'].shape[1], data_eval['y'].shape[1]
        
        # Treatment is Continuous
        if self.t_dim == 1:
            data_eval_temp = copy.deepcopy(data_eval)
            data_eval_temp['t'] = np.zeros(shape = [self.n, self.t_dim])
            self.log.log('    treatment为连续数值型, 则略过 计算转化率 和 Qini Curve')
            y_hat = model.pred(data_eval, target = 'y') 
            y_hat_0 =  model.pred(data_eval_temp, target = 'y')
            mse_difference = get_mse(y_hat - y_hat_0, data_eval['y']-data_eval['mu_0'])
            return {'mse_difference': mse_difference}

        ## Treatment is Multi-calss: encoded as one-hot encoder
        ## if self.t_dim > 1: (此处可以省略，因为)
        data_eval_temp = copy.deepcopy(data_eval)
        conversion = np.zeros(shape = [self.n, self.t_dim])
        for k in range(self.t_dim):
            # set the k-th column of t as 1
            data_eval_temp['t'] = np.zeros(shape = [self.n, self.t_dim])
            data_eval_temp['t'][:,k] = 1
            # predicted prob of Y (n*self.y_dim)
            # ATTENTION: if use loss_type = sigmoid, the returned PROB may not sum as ONE, may cause issues in evaluation
            #            but if we only cares about CONVERSION, just calculate conversion=1-P(Y=0)
            conversion[:,k] = 1-(model.pred(data_eval_temp, target = 'y'))[:,0]
        
        # Best Treatment and Predicted Conversion under observed treatment
        data_eval['best_treat'] = np.argmax(conversion, axis=1) # denote best_t=0,...,t_dim-1
        data_eval['exact_conversion'] = (conversion * data_eval['t']).sum(axis=1)
        data_eval['t_scalar'] = np.argmax(data_eval['t'], axis=1) # denote t=0,...,t_dim-1
        data_eval['y_scalar'] = 1-data_eval['y'][:,0] # denote convert (1) and not convert (0)
        # Propensity Score ( P(T=t): probability of receiving current treatment )
        self.log.log('\n=== 计算每个样本点的propensity score:')
        data_eval['ps_t'] = (model.pred(data_eval, target = 'treatment') * data_eval['t']).sum(axis=1)
        
        # Save Detailed Results
        res_df = pd.DataFrame(np.concatenate((
                            data_eval['t'].reshape([-1,self.t_dim]),
                            data_eval['y'].reshape([-1,self.y_dim]), 
                            data_eval['t_scalar'].reshape([-1,1]),
                            data_eval['y_scalar'].reshape([-1,1]),
                            data_eval['best_treat'].reshape([-1,1]),
                            data_eval['ps_t'].reshape([-1,1]), 
                            data_eval['exact_conversion'].reshape([-1,1]),                     
                            conversion),axis=1), \
                            columns=['t_'+str(k) for k in range(self.t_dim)] + \
                                    ['y_'+str(k) for k in range(self.y_dim)] + \
                                    ['t_scalar', 'y_scalar','best_treat', 'ps_t', 'exact_conversion'] + \
                                    ['conv_'+str(k) for k in range(self.t_dim)])
        if save_detailed_results_to_csv:
            file_name = os.path.join(FLAGS.output_dir,'model_evaluate','Evaluate_details_{}_{}_exp_{}.csv'.format(model_name, FLAGS.config_name, num_experiment))
            res_df.to_csv(file_name, index=None)
            self.log.log('\n=== 保存详细结果在 {}'.format(file_name))
            
        
        
        # Qini Curve
        self.log.log('\n=== 保存可视化qini curve和barplot:')
        # pair the Control Group with Each Treatment Group, respectively
        if self.t_dim > 2:
            qini_score_list = []
        for k in range(1, self.t_dim):
            # Filter Data: select the records with Treatment = 0 or k 
            res_df_temp = res_df.loc[(res_df.iloc[:,0]==1)|(res_df.iloc[:,k]==1),:].copy()
            # Function plot_uplift only uses ['t','y','uplift']
            res_df_temp['t'] = res_df_temp['t_' + str(k)]
            res_df_temp['uplift'] = res_df_temp['conv_' + str(k)] - res_df_temp['conv_' + str(0)]
            res_df_temp['y'] = 1-res_df_temp['y_0'] # only cares about conversion (不分期=0, 分期=1, 而不关注分期类型)
            output_path=os.path.join(FLAGS.output_dir, 'model_evaluate', 'Evaluate_plot_{type}_{model_name}_{config_name}_treat{k}_exp_{r}.png'.format(type='{type}', model_name=model_name, config_name=FLAGS.config_name, k=k, r=num_experiment))
            # TODO: 保存res_df_temp
            res_df_temp.to_csv(os.path.join(FLAGS.output_dir, 'model_evaluate', 'Data_for_qini_{model_name}_{config_name}_exp_{r}.csv'.format(type='{type}', model_name=model_name, config_name=FLAGS.config_name, r=num_experiment)))
            # 画qini curve图
            plot_uplift(res_df_temp, output_path=output_path, model_name=model_name)
        
            # 此处的qini score由sklift.metrics.qini_auc_score计算
            qini_score = qini_auc_score(y_true=res_df_temp['y'], uplift=res_df_temp['uplift'],treatment=res_df_temp['t'])
            # 如果类别>2, 需要保留所有实验组相对于对照组计算的qini_score
            if self.t_dim > 2:
                qini_score_list.append(qini_score)
        

        # 模型策略的平均转化率
        ## sample size
        n = res_df.shape[0]
        ## res_cross[k] = #{ best_treatment=observed treatment=k 的人群}
        res_cross= {}
        for k in range(self.t_dim):
            res_cross[k] = res_df.loc[(res_df['t_scalar']==res_df['best_treat'])&(res_df['t_scalar']==k),:]
        ## n_model[k] = #{ best_treatment=k 的人群 }
        n_model = {}
        for k in range(self.t_dim):
            n_model[k] = res_df.loc[res_df['best_treat']==k,:].shape[0]
        ## calculate the Average Conversion Rate
        conv_rate_model = 0
        for k in range(self.t_dim):
            ### to avoid {best_treatment=observed treatment=k} = empty set
            if len(res_cross[k])==0:
                self.log.log('For treatment {}: no observed overlapping sample (best=observed={})'.format(k, k))
            elif len(res_cross[k])>0:
                conv_rate_model += n_model[k]/n * res_cross[k]['y_scalar'].mean()
        conv_rate_random = res_df['y_scalar'].mean()
        conv_rate = pd.DataFrame.from_dict({'conv_rate_model': [conv_rate_model], 
                                            'conv_rate_random': [conv_rate_random],
                                            'qini_score': [qini_score]})
        log.log('=== 平均转化率：模型策略 v.s. 随机结果; 以及Qini Score')
        log.log(str(conv_rate))

        if save_results_to_csv:
            conv_rate.to_csv(os.path.join(FLAGS.output_dir,'model_evaluate','Evaluate_summary_rank_{}_{}_exp_{}.csv'.format(model_name, FLAGS.config_name, num_experiment)), index=None)
        
        if self.t_dim == 2:
            return {'qini_score': qini_score, 'conv_rate_model': conv_rate_model, 'conv_rate_random': conv_rate_random}
        elif self.t_dim > 2:
            return {'qini_score_list': ';'.join([str(s) for s in qini_score_list]), 'conv_rate_model': conv_rate_model, 'conv_rate_random': conv_rate_random}






if __name__ == '__main__':

    # log
    log = utils.Log(log_single=os.path.join(FLAGS.output_dir,'logs'), res = FLAGS.config_name)

    # data for training
    D_load = utils.Load_Data([63, 27, 10])
    D_load.load(FLAGS.data_path)

    # file path for save evaluation results
    file_name_summary_eval = os.path.join(FLAGS.output_dir,'model_evaluate','Evaluate_summary_all_{model}_{config}.csv'.format(model = FLAGS.model_name, config = FLAGS.config_name))
        
    
    # Run Replicated Experiment for num_experiments times for DER
    params_full = get_str_from_FLAGS(FLAGS,excluded = {'config_name'})
    log.log('== Parameters: config_name={}\n'.format(FLAGS.config_name) + params_full)
    for r in range(FLAGS.num_experiments):
        log.log('==================================================')
        log.log('== {r}-th experiment of {num_experiments} experiments\n'.format(r=r, num_experiments = FLAGS.num_experiments) + params_full)    
        log.log('==================================================')


        # shuffle data and split into D_load.train/valid/test
        D_load.split_data()
        D_load.shuffle()
        

        # data for evaluate (somtimes, evaluation uses RCT, modeling uses Observational)
        D_load_eval = D_load
        # DER and ADR (ours) used the same Module (module_DER_exxtended.PY)
        if FLAGS.model_name == 'DER' or FLAGS.model_name == 'ADR':
            print("choose DER or ADR!!!") 
            model_name = FLAGS.model_name
            # build model 
            model = RepModel_DER(D_load.train, D_load.valid, log, model_name=model_name, num_experiment=r)
        elif FLAGS.model_name == 'DR':
            print("choose DR!!!") 
            model_name = 'DR'
            model = RepModel_DR(D_load.train, D_load.valid, log, model_name=model_name, num_experiment=r)

        
        ############ validate model
        w_I_true, w_I_false, w_C_true, w_C_false, w_A_true, w_A_false = model.validate_representation_learning(num_experiment=r)
        # model_DER.save_rep(D_load.valid, data_name='valid_data')
        file_name_summary_validate = os.path.join(FLAGS.output_dir,'model_validate','Rep_w_all_{model}_{config}.csv'.format(model =model_name, config = FLAGS.config_name))
        if r==0:
            with open(file_name_summary_validate, 'w') as f:
                f.write('num_experiments, w_I_true, w_I_false, w_C_true, w_C_false, w_A_true, w_A_false\n')
        with open(file_name_summary_validate, 'a') as f:
            w = ','.join([str(w) for w in [w_I_true, w_I_false, w_C_true, w_C_false, w_A_true, w_A_false]])
            s = '{num_experiments}, {w}\n'.format(num_experiments=r, w=w)
            f.write(s)


        ############ evaluate model
        evaluator = Evaluator(log) 
        eval_res_by_factual_response = evaluator.evaluate_by_factual_response(model, D_load_eval.test, model_name=model_name, num_experiment=r)
        eval_res_by_uplift = evaluator.evaluate_by_uplift(model, D_load_eval.test, model_name=model_name, num_experiment=r)
        # 将每次的experiment统一保存到file_name_summary_eval
        if r==0:
            with open(file_name_summary_eval, 'w') as f:
                res_names = ','.join([','.join(eval_res_by_factual_response.keys()), ','.join(eval_res_by_uplift.keys())])
                f.write('num_experiments, {res_names}\n'.format(res_names=res_names))
        with open(file_name_summary_eval, 'a') as f:
            log.log('=======write into evaluate_summary_all')
            res_values = ','.join([','.join([str(v) for v in eval_res_by_factual_response.values()]), ','.join([str(v) for v in eval_res_by_uplift.values()])])
            f.write('{num_experiments}, {res_values}\n'.format(num_experiments=r, res_values=res_values))

    
    
