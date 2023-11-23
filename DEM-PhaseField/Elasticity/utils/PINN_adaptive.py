# Class file for Physics Informed Neural Network

#import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow as tf 
import numpy as np
import time

class Elasticity2D:
    # Initialize the class
    def __init__(self, model_data, X_bnd, NN_param, X_f):
        
        #initialize fields for model data
        self.E = model_data['E']                        #model_dataはメイン85行目
        self.nu = model_data['nu']
        
        #plane stress        平面応力の弾性定数
        self.c11 = self.E/(1-self.nu**2)                
        self.c22 = self.E/(1-self.nu**2)
        self.c12 = self.E*self.nu/(1-self.nu**2)
        self.c21 = self.E*self.nu/(1-self.nu**2)
        self.c31 = 0.0
        self.c32 = 0.0
        self.c13 = 0.0
        self.c23 = 0.0
        self.c33 = self.E/(2*(1+self.nu))

        
        self.x_bnd = X_bnd[:, 0:1]              #physX
        self.y_bnd = X_bnd[:, 1:2]              #physY
        self.wt_bnd = X_bnd[:, 2:3]             #重み
        self.trac_x_bnd = X_bnd[:, 3:4]         #トラクションx
        self.trac_y_bnd = X_bnd[:, 4:5]         #トラクションy      
        
        self.lb = model_data['lb'] #lower bound of the plate
        self.ub = model_data['ub']
        
        self.layers = NN_param['layers']                                    #NNのレイヤー構造を指定
        self.data_type = NN_param['data_type']                              #TFのデータ型を指定
        self.weights, self.biases = self.initialize_NN(self.layers)         #NNを初期化

        '''        
        self.model_parameters = [self.weights, self.biases]                          #モデル内のパラメータを取得

        #print(self.weights)
        #print(self.biases)
        
        # 重みとバイアスを平坦化
        self.flat_weights = tf.concat([tf.reshape(w, [-1]) for w in self.weights], axis=0)
        self.flat_biases = tf.concat([tf.reshape(b, [-1]) for b in self.biases], axis=0)

        # 重みとバイアスを結合
        self.initial_position = tf.concat([self.flat_weights, self.flat_biases], axis=0)
        '''


        # tf Placeholders
        '''
        self.x_int_tf = tf.compat.v1.placeholder(self.data_type)            #積分点について
        self.y_int_tf = tf.compat.v1.placeholder(self.data_type)              
        self.wt_int_tf = tf.compat.v1.placeholder(self.data_type)             
        
        self.x_bnd_tf = tf.compat.v1.placeholder(self.data_type)             #physX 
        self.y_bnd_tf = tf.compat.v1.placeholder(self.data_type)             #physY
        self.wt_bnd_tf = tf.compat.v1.placeholder(self.data_type)            #重み
        self.trac_x_bnd_tf = tf.compat.v1.placeholder(self.data_type)        #トラクションX
        self.trac_y_bnd_tf = tf.compat.v1.placeholder(self.data_type)        #トラクションY
        '''
        self.x_bnd_tf = self.x_bnd           #physX
        self.y_bnd_tf = self.y_bnd           #physY
        self.wt_bnd_tf = self.wt_bnd         #重み
        self.trac_x_bnd_tf = self.trac_x_bnd     #トラクションx
        self.trac_y_bnd_tf = self.trac_y_bnd     #トラクションy              
        self.x_int_tf = X_f[:,0:1]       #積分点x
        self.y_int_tf = X_f[:,1:2]       #積分点y
        self.wt_int_tf = X_f[:,2:3]
       

        # tf Graphs
        self.u_pred, self.v_pred = self.net_uv(self.x_int_tf,self.y_int_tf)
        self.fx_pred, self.fy_pred = self.net_traction(self.x_bnd_tf, self.y_bnd_tf,                                               
                                               self.trac_x_bnd_tf, self.trac_y_bnd_tf)
        self.energy_pred, self.sigma_x_pred, self.sigma_y_pred, \
              self.tau_xy_pred = self.net_energy(self.x_int_tf,self.y_int_tf)
              
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_int_tf,self.y_int_tf)      

        # Loss
        self.loss_neu = tf.reduce_sum((self.fx_pred+self.fy_pred)*self.wt_bnd)              #境界条件
        self.loss_int = tf.reduce_sum(self.energy_pred*self.wt_int_tf)                      #内部エネルギー
        
        self.loss = self.loss_int  - self.loss_neu       

        #reduce_sumは配列内の全ての要素を足し算する関数                             

        # Optimizers
        '''
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,                                              #最小化する損失関数
                                                                method='L-BFGS-B',                                      #
                                                                options={'maxiter': 50,                                 #最大反復回数
                                                                         'maxfun': 50,                                  #最大関数評価回数
                                                                         'maxcor': 100,                                 #最大コリンズ数
                                                                         'maxls': 50,                                   #最大ラインサーチ数
                                                                         'ftol': 1.0 * np.finfo(float).eps})            #収束基準
        '''
        self.optimizer = tfp.optimizer.lbfgs_minimize(self.loss,
                                                      initial_position = self.initialize_NN(self.layers),
                                                      max_iterations = 50,                                              #最大反復回数
                                                      num_correction_pairs = 100,                                       #最大コリンズ数
                                                      max_line_search_iterations = 50,                                  #最大ラインサーチ数
                                                      f_relative_tolerance = 1.0 * np.finfo(float).eps                  #収束基準
                                                      )
    
        self.lbfgs_buffer = []
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        #self.optimizerを使用してモデルを訓練し、seld.optimizer_Adamを用いてモデルをさらに微調整することができる

        # tf session　セッションの初期化。TFグラフの実行環境を提供。
        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self,layers):
        '''
        重み、バイアスを初期化する
        '''
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])                                           #重み行列を初期化　lとl+1はそれぞれ現在の層と次の層のノード数
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.data_type), dtype=self.data_type)       #バイアスを初期化　バイアスのサイズはl+1(次の層)のノード数に依存している
            weights.append(W)       #リストに追加
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        '''
        重み行列を初期化する
        '''
        in_dim = size[0]            #前の層のノード数(入力次元数)
        out_dim = size[1]           #次の層のノード数(出力次元数)
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))       #標準偏差を計算　Xavierの初期化法において必要
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.data_type), dtype=self.data_type)

    def neural_net(self,X,weights,biases):
        '''
        隠れ層と最終層の順伝播(forward pass)を計算
        '''
        num_layers = len(weights) + 1                               #NNの層の数
		
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0         #入力データのスケーリング？？？
        for l in range(0, num_layers - 2):                          #隠れ層の計算
            W = weights[l]
            b = biases[l]
            H = tf.nn.relu(tf.add(tf.matmul(tf.cast(H, W.dtype), W), b))**2           #matmulでHとWをかける   addでbと足し合わせる　活性化関数の出力を2乗する
        W = weights[-1]                                             #最終層の重み
        b = biases[-1]                                              #最終層のバイアス
        Y = tf.add(tf.matmul(H, W), b)                              #最終層の計算
        return Y                                                    
    
    def net_f_uv(self,x,y):
        '''
        入力に対する偏微分を計算してfを返す
        '''

        x = tf.Variable(x)
        y = tf.Variable(y)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)

            u, v = self.net_uv(x, y)
        u_x = tape.gradient(u, x)
        u_xx = tape.gradient(u_x, x)
        v_y = tape.gradient(v, y)
        v_yx = tape.gradient(v_y, x)
        u_y = tape.gradient(u, y)
        u_yy = tape.gradient(u_y, y)
        v_x = tape.gradient(v, x)
        v_xy = tape.gradient(v_x, y)
        
        f_u = -(self.c11 * u_xx + self.c12 * v_yx + self.c33 * u_yy + self.c33 * v_xy)

        u_xy = tape.gradient(u_x, y)
        v_yy = tape.gradient(v_y, y)
        u_yx = tape.gradient(u_y, x)
        v_xx = tape.gradient(v_x, x)
        
        # 不要になった tape を削除
        del tape

        f_v = -(self.c21 * u_xy + self.c22 * v_yy + self.c33 * u_yx + self.c33 * v_xx)
          
        return f_u, f_v

        
        '''
        u, v = self.net_uv(x,y)                                     #NNの入力xyから出力uvを計算
        u_x = tf.gradients(u,x)[0]                                  
        u_xx = tf.gradients(u_x,x)[0]
        v_y = tf.gradients(v,y)[0]
        v_yx = tf.gradients(v_y,x)[0]
        u_y = tf.gradients(u,y)[0]
        u_yy = tf.gradients(u_y,y)[0]
        v_x = tf.gradients(v,x)[0]
        v_xy = tf.gradients(v_x,y)[0]

        f_u = -(self.c11*u_xx + self.c12*v_yx + self.c33*u_yy + self.c33*v_xy)

        u_xy = tf.gradients(u_x,y)[0]
        v_yy = tf.gradients(v_y,y)[0]
        u_yx = tf.gradients(u_y,x)[0]
        v_xx = tf.gradients(v_x,x)[0]

        f_v = -(self.c21*u_xy + self.c22*v_yy + self.c33*u_yx + self.c33*v_xx)
        
        return f_u, f_v
        '''

    def net_uv(self,x,y):
        '''
        ニューラルネットワークを使用して入力から出力を求める
        '''

        X = tf.concat([x.value(),y.value()],1)                                  #1はaxis

        uv = self.neural_net(X,self.weights,self.biases)        #NNを介してuv（求めたい値）　解候補　を計算
        u = uv[:,0:1]
        v = uv[:,1:2]

        return u, v

    def net_traction(self,x,y,tracX,tracY):     #tracXYはトラクション成分（？）
        '''
        トラクションを返す
        '''

        u, v = self.net_uv(x,y)
        
        trX = u*tracX       #トラクション
        trY = v*tracY
        
        return trX, trY
    
    def net_energy(self,x,y):
        '''
        弾性ひずみエネルギー密度を計算
        '''
        x = tf.Variable(x)
        y = tf.Variable(y)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)

            u, v = self.net_uv(x, y)
        u_x = tape.gradient(u, x)[0]
        u_y = tape.gradient(u, y)[0]
        v_y = tape.gradient(v, y)[0]
        v_x = tape.gradient(v, x)[0]
        # 不要になった tape を削除
        del tape

        u_xy = u_y + v_x

        sigmaX = self.c11 * u_x + self.c12 * v_y
        sigmaY = self.c21 * u_x + self.c22 * v_y
        tauXY = self.c33 * u_xy

        energy = 0.5 * (sigmaX * u_x + sigmaY * v_y + tauXY * u_xy)

        return energy, sigmaX, sigmaY, tauXY

        '''
        with tf.GradientTape() as tape:
            u, v = self.net_uv(x,y)
            u_x = tape.gradient(u,x)[0]
        
        with tf.GradientTape() as tape:
            u, v = self.net_uv(x,y)
            u_y = tape.gradient(u,y)[0]
        
        with tf.GradientTape() as tape:
            u, v = self.net_uv(x,y)
            v_y = tape.gradient(v,y)[0]
        
        with tf.GradientTape() as tape:
            u, v = self.net_uv(x,y)
            v_x = tape.gradient(v,x)[0]
        
        u_xy = u_y + v_x
        
        sigmaX = self.c11*u_x + self.c12*v_y            #教科書p.160
        sigmaY = self.c21*u_x + self.c22*v_y
        tauXY = self.c33*u_xy
        
        energy = 0.5*(sigmaX*u_x + sigmaY*v_y + tauXY*u_xy)         #教科書p.156
        
        return energy, sigmaX, sigmaY, tauXY
        '''

    def callback(self, loss):
        '''
        最新の損失値'loss'を表示する。最適化の進行状況を可視化。
        '''
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)      #'lbfgs_buffer'と呼ばれるnumpy配列に最新の損失値'loss'を追加する
        print('Loss:', loss)

    def train(self, X_f, nIter):
        '''
        ニューラルネットワークモデルを学習する
        '''
        
        tf_dict = {self.x_bnd_tf: self.x_bnd,           #physX
                   self.y_bnd_tf: self.y_bnd,           #physY
                   self.wt_bnd_tf: self.wt_bnd,         #重み
                   self.trac_x_bnd_tf: self.trac_x_bnd,     #トラクションx
                   self.trac_y_bnd_tf: self.trac_y_bnd,     #トラクションy              
                   self.x_int_tf: X_f[:,0:1],       #積分点x
                   self.y_int_tf: X_f[:,1:2],       #積分点y
                   self.wt_int_tf: X_f[:,2:3]}      #重み

        start_time = time.time()
        self.loss_adam_buff = np.zeros(nIter)       #nIterは反復回数
        
        for it in range(nIter):
            
            self.sess.run(self.train_op_Adam, tf_dict)                                                      #フィードデータtf_dictを用いてトレーニングオペレーション（最小化）をする
            loss_value = self.sess.run(self.loss, tf_dict)                                                  #現在の損失値
            self.loss_adam_buff[it] = loss_value                                                            #配列に記録
            # Print
            if it % 10 == 0:                                                                                #反復回数が10の倍数の場合に進捗状況を表示
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_neu = self.sess.run(self.loss_neu, tf_dict)
                loss_int = self.sess.run(self.loss_int, tf_dict)
                print('It: %d, Total Loss: %.3e, Int Loss: %.3e, Neumann Loss: %.3e, Time: %.2f' %
                      (it, loss_value, loss_int, loss_neu, elapsed))                                        #反復回数、総合損失、内部損失？？、ノイマン損失？？、実行時間
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)                                                #最適化      
        
    def predict(self, X_star):
        '''
        与えられたデータ X_star に対して予測を行う
        '''
        tf_dict = {self.x_int_tf: X_star[:,0:1], self.y_int_tf: X_star[:,1:2]}                              #feed dictionaryは 206行目付近　積分点xy座標
                                                                                                            
        u_star = self.sess.run(self.u_pred, tf_dict)                                                        #uの予測値
        v_star = self.sess.run(self.v_pred, tf_dict)                                                        #vの予測値
        energy_star = self.sess.run(self.energy_pred, tf_dict)                                              #エネルギーの予測値
        sigma_x_star = self.sess.run(self.sigma_x_pred, tf_dict)                                            #σ_xの予測値
        sigma_y_star = self.sess.run(self.sigma_y_pred, tf_dict)                                            #σ_yの予測値
        tau_xy_star = self.sess.run(self.tau_xy_pred, tf_dict)                                              #τ_xyの予測値

        return u_star, v_star, energy_star, sigma_x_star, sigma_y_star, tau_xy_star
    
    def predict_f(self, X_star):
        '''
        与えられたデータ X_star に対して予測を行う fffffffffffff errorなのか？メイン156行目
        '''
        tf_dict = {self.x_int_tf: X_star[:,0:1], self.y_int_tf: X_star[:,1:2]}                       
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
        
        return f_u_star, f_v_star
    
class Elasticity3D:
    # Initialize the class
    def __init__(self, model_data, model_pts, NN_param):
        
        #initialize fields for model data
        self.E = model_data['E']
        self.nu = model_data['nu']
        
        self.c11 = self.E *(1-self.nu)/((1+self.nu)*(1-2*self.nu))
        self.c22 = self.E *(1-self.nu)/((1+self.nu)*(1-2*self.nu))
        self.c33 = self.E *(1-self.nu)/((1+self.nu)*(1-2*self.nu))
        self.c12 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c13 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c21 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c23 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c31 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c32 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c44 = self.E/(2*(1+self.nu))
        self.c55 = self.E/(2*(1+self.nu))
        self.c66 = self.E/(2*(1+self.nu))
        #self.c13 = 0.0
        #self.c23 = 0.0        
        
        #initialized fields for model points
        X_int = model_pts['X_int']
        X_bnd = model_pts['X_bnd']

        self.x_int = X_int[:, 0:1]
        self.y_int = X_int[:, 1:2]
        self.z_int = X_int[:, 2:3]
        self.wt_int = X_int[:, 3:4]
        
        self.x_bnd = X_bnd[:, 0:1]
        self.y_bnd = X_bnd[:, 1:2]
        self.z_bnd = X_bnd[:, 2:3]
        self.wt_bnd = X_bnd[:, 3:4]
        self.trac_x_bnd = X_bnd[:, 4:5]
        self.trac_y_bnd = X_bnd[:, 5:6]
        self.trac_z_bnd = X_bnd[:, 6:7]
        
        #tofix: use the values set in the subclass?
        self.lb = np.array([np.min(self.x_int), np.min(self.y_int), np.min(self.z_int)])
        self.ub = np.array([np.max(self.x_int), np.max(self.y_int), np.max(self.z_int)])
        
        self.layers = NN_param['layers']
        self.data_type = NN_param['data_type']
        self.weights, self.biases = self.initialize_NN(self.layers)

        # tf Placeholders
        self.x_int_tf = tf.placeholder(self.data_type)
        self.y_int_tf = tf.placeholder(self.data_type)
        self.z_int_tf = tf.placeholder(self.data_type)
        self.wt_int_tf = tf.placeholder(self.data_type)
        
        self.x_bnd_tf = tf.placeholder(self.data_type)
        self.y_bnd_tf = tf.placeholder(self.data_type)
        self.z_bnd_tf = tf.placeholder(self.data_type)
        self.wt_bnd_tf = tf.placeholder(self.data_type)
        self.trac_x_bnd_tf = tf.placeholder(self.data_type)
        self.trac_y_bnd_tf = tf.placeholder(self.data_type)
        self.trac_z_bnd_tf = tf.placeholder(self.data_type)

        # tf Graphs
        self.u_pred, self.v_pred, self.w_pred = self.net_uvw(self.x_int_tf, 
                                                             self.y_int_tf,
                                                             self.z_int_tf)
        self.fx_pred, self.fy_pred, self.fz_pred = self.net_traction(self.x_bnd_tf, 
                           self.y_bnd_tf, self.z_bnd_tf, self.trac_x_bnd_tf, 
                           self.trac_y_bnd_tf, self.trac_z_bnd_tf)
        self.energy_pred, self.sigma_x_pred, self.sigma_y_pred, \
                self.sigma_z_pred, self.tau_xy_pred, self.tau_yz_pred, \
                self.tau_zx_pred = self.net_energy(self.x_int_tf, self.y_int_tf, self.z_int_tf)
                
        self.f_u_pred, self.f_v_pred, self.f_w_pred = self.net_f_uvw(self.x_int_tf, \
                                                    self.y_int_tf, self.z_int_tf)  

        # Loss
        self.loss_neu = tf.reduce_sum((self.fx_pred+self.fy_pred+self.fz_pred)*self.wt_bnd)
        self.loss_int = tf.reduce_sum(self.energy_pred*self.wt_int_tf)
        
        self.loss = self.loss_int  - self.loss_neu                                    

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 100,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
    
        self.lbfgs_buffer = []
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self,layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.data_type), dtype=self.data_type)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.data_type), dtype=self.data_type)

    def neural_net(self,X,weights,biases):
        num_layers = len(weights) + 1
		
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        #H = tf.sigmoid(tf.add(tf.matmul(H, weights[0]), biases[0]))
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            #H = tf.tanh(tf.add(tf.matmul(H, W), b))
            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))**2
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_f_uvw(self,x,y,z):

        u, v, w = self.net_uvw(x, y, z)
        u_x = tf.gradients(u,x)[0]
        u_y = tf.gradients(u,y)[0]
        u_z = tf.gradients(u,z)[0]
        v_x = tf.gradients(v,x)[0]
        v_y = tf.gradients(v,y)[0]
        v_z = tf.gradients(v,z)[0]
        w_x = tf.gradients(w,x)[0]
        w_y = tf.gradients(w,y)[0]
        w_z = tf.gradients(w,z)[0]
        gamma_xy = u_y + v_x
        gamma_yz = v_z + w_y
        gamma_zx = u_z + w_x
        
        sigmaX = self.c11*u_x + self.c12*v_y + self.c13*w_z
        sigmaY = self.c21*u_x + self.c22*v_y + self.c23*w_z
        sigmaZ = self.c31*u_x + self.c32*v_y + self.c33*w_z
        tauXY = self.c44*gamma_xy
        tauYZ = self.c55*gamma_yz
        tauZX = self.c66*gamma_zx
        
        f_u = -(tf.gradients(sigmaX, x)[0] + tf.gradients(tauXY, y)[0] + \
                tf.gradients(tauZX, z)[0])
        f_v = -(tf.gradients(tauXY, x)[0] + tf.gradients(sigmaY, y)[0] + \
                tf.gradients(tauYZ, z)[0])
        f_w = -(tf.gradients(tauZX, x)[0] + tf.gradients(tauYZ, y)[0] + \
                tf.gradients(sigmaZ, z)[0])
                
        return f_u, f_v, f_w


    def net_uvw(self,x,y):

        X = tf.concat([x,y],1)

        uvw = self.neural_net(X,self.weights,self.biases)
        
        u = uvw[:,0:1]
        v = uvw[:,1:2]
        w = uvw[:,2:3]

        return u, v, w

    def net_traction(self,x,y,z,tracX,tracY,tracZ):

        u, v, w = self.net_uvw(x,y,z)
        
        trX = u*tracX
        trY = v*tracY
        trZ = w*tracZ
        
        return trX, trY, trZ
    
    def net_energy(self, x, y, z):
        u, v, w = self.net_uvw(x, y, z)
               
        u_x = tf.gradients(u,x)[0]
        u_y = tf.gradients(u,y)[0]
        u_z = tf.gradients(u,z)[0]
        v_x = tf.gradients(v,x)[0]
        v_y = tf.gradients(v,y)[0]
        v_z = tf.gradients(v,z)[0]
        w_x = tf.gradients(w,x)[0]
        w_y = tf.gradients(w,y)[0]
        w_z = tf.gradients(w,z)[0]
        gamma_xy = u_y + v_x
        gamma_yz = v_z + w_y
        gamma_zx = u_z + w_x
        
        sigmaX = self.c11*u_x + self.c12*v_y + self.c13*w_z
        sigmaY = self.c21*u_x + self.c22*v_y + self.c23*w_z
        sigmaZ = self.c31*u_x + self.c32*v_y + self.c33*w_z
        tauXY = self.c44*gamma_xy
        tauYZ = self.c55*gamma_yz
        tauZX = self.c66*gamma_zx
        
        
        energy = 0.5*(sigmaX*u_x + sigmaY*v_y + sigmaZ*w_z + tauYZ*gamma_yz + \
                          tauZX*gamma_zx + tauXY*gamma_xy)
        
        return energy, sigmaX, sigmaY, sigmaZ, tauXY, tauYZ, tauZX    
        
    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
        print('Loss:', loss)

    def train(self, X_f, nIter):
        
        tf_dict = {self.x_bnd_tf: self.x_bnd, 
                   self.y_bnd_tf: self.y_bnd,
                   self.z_bnd_tf: self.z_bnd,
                   self.wt_bnd_tf: self.wt_bnd,
                   self.trac_x_bnd_tf: self.trac_x_bnd,
                   self.trac_y_bnd_tf: self.trac_y_bnd,
                   self.trac_z_bnd_tf: self.trac_z_bnd,
                   self.x_int_tf: X_f[:,0:1],
                   self.y_int_tf: X_f[:,1:2],
                   self.z_int_tf: X_f[:,2:3],
                   self.wt_int_tf: X_f[:,3:4]}

        start_time = time.time()
        self.loss_adam_buff = np.zeros(nIter)
        
        for it in range(nIter):
            
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_adam_buff[it] = loss_value
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_neu = self.sess.run(self.loss_neu, tf_dict)
                loss_int = self.sess.run(self.loss_int, tf_dict)
                print('It: %d, Total Loss: %.3e, Int Loss: %.3e, Neumann Loss: %.3e, Time: %.2f' %
                      (it, loss_value, loss_int, loss_neu, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)            
        
    def predict(self, X_star):

        tf_dict = {self.x_int_tf: X_star[:,0:1], self.y_int_tf: X_star[:,1:2], 
                   self.z_int_tf: X_star[:,2:3]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        sigma_x_star = self.sess.run(self.sigma_x_pred, tf_dict)
        sigma_y_star = self.sess.run(self.sigma_y_pred, tf_dict)
        sigma_z_star = self.sess.run(self.sigma_z_pred, tf_dict)
        tau_xy_star = self.sess.run(self.tau_xy_pred, tf_dict)
        tau_yz_star = self.sess.run(self.tau_yz_pred, tf_dict)
        tau_zx_star = self.sess.run(self.tau_zx_pred, tf_dict)
                
        energy_star = self.sess.run(self.energy_pred, tf_dict)

        return u_star, v_star, w_star, energy_star, sigma_x_star, sigma_y_star, \
            sigma_z_star, tau_xy_star, tau_yz_star, tau_zx_star
            
    def predict_f(self, X_star):
        
        tf_dict = {self.x_int_tf: X_star[:,0:1], self.y_int_tf: X_star[:,1:2],
                   self.z_int_tf: X_star[:,2:3]}                       
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
        f_w_star = self.sess.run(self.f_w_pred, tf_dict)
        return f_u_star, f_v_star, f_w_star
                