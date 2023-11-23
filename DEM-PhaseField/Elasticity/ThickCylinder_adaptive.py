# Implements the 2D pressurized cylinder benchmark problem subjected to 
# internal pressure on the inner circular edge, under plane stress condition
#commit test
#import tensorflow as tf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import time
import os
from utils.gridPlot2D import getExactDisplacements
from utils.gridPlot2D import plotDeformedDisp
from utils.gridPlot2D import energyError
from utils.gridPlot2D import scatterPlot
from utils.gridPlot2D import createFolder
from utils.gridPlot2D import energyPlot
from utils.gridPlot2D import refineElemVertex
tf.compat.v1.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

from utils.Geom import Geometry2D
from utils.PINN_adaptive import Elasticity2D

class Annulus(Geometry2D):              #Geometry2Dを継承
    '''
     Class for definining a quarter-annulus domain centered at the orgin
         (the domain is in the first quadrant)
     Input: rad_int, rad_ext - internal and external radii of the annulus
    '''
    def __init__(self, radInt, radExt):            
        
        geomData = dict()
        # Set degrees　次数
        geomData['degree_u'] = 1
        geomData['degree_v'] = 2
        
        # Set control points　CP数
        geomData['ctrlpts_size_u'] = 2
        geomData['ctrlpts_size_v'] = 3
                
        geomData['ctrlpts'] = [[radInt,0.,0.],
                    [radInt*np.sqrt(2)/2, radInt*np.sqrt(2)/2, 0.],
                    [0., radInt, 0.],
                    [radExt, 0., 0.],
                    [radExt*np.sqrt(2)/2, radExt*np.sqrt(2)/2, 0.],
                    [0., radExt, 0.]]
        
        geomData['weights'] = [1, np.sqrt(2)/2, 1, 1, np.sqrt(2)/2, 1]
        
        # Set knot vectors  (n_cp + p + 1)
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_v'] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        super().__init__(geomData)      #継承元のコンストラクタを上書きする

class PINN_TC(Elasticity2D):            #Elasticity2Dを継承しつつ、新しい機能を追加できるようになる
    '''
    Class including (symmetry) boundary conditions for the thick cylinder problem
    '''
    def __init__(self, model_data, xEdgePts, NN_param, X_f):
        
        super().__init__(model_data,xEdgePts, NN_param, X_f)
        
    def net_uv(self, x, y):             

        X = tf.concat([x, y], 1)      

        uv = self.neural_net(X,self.weights,self.biases)

        uv = tf.cast(uv, dtype=tf.double)

        u = x*uv[:, 0:1]
        v = y*uv[:, 1:2]

        return u, v

if __name__ == "__main__":
    
    originalDir = os.getcwd()                                           #current directory
    foldername = 'ThickCylinder'    
    createFolder('./'+ foldername + '/')                                #フォルダを作る
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))         #ディレクトリを移動
    figHeight = 6
    figWidth = 8
    
    model_data = dict()
    model_data['E'] = 1e5
    model_data['nu'] = 0.3
    
    model = dict()
    model['E'] = 1e5    #ヤング率
    model['nu'] = 0.3   #ポアソン比
    model['radInt'] = 1.0   #内径
    model['radExt'] = 4.0   #外径
    model['P'] = 10.0   #内圧
    
    # Domain bounds
    model_data['lb'] = np.array([0.0,0.0]) #lower bound of the plate
    model_data['ub'] = np.array([model['radExt'],model['radExt']]) # Upper bound of the plate
    
    NN_param = dict()
    NN_param['layers'] = [2, 30, 30, 30, 2]
    NN_param['data_type'] = tf.float32
    
    # Generating points inside the domain using GeometryIGA
    myAnnulus = Annulus(model['radInt'], model['radExt'])
    
    numElemU = 20       #u方向要素数
    numElemV = 20       #v方向要素数
    numGauss = 2        #uv方向のガウス点数
    
    vertex = myAnnulus.genElemList(numElemU, numElemV)                          #vertexは四角形の頂点座標
    xPhys, yPhys, wgtsPhys = myAnnulus.getElemIntPts(vertex, numGauss)          #積分点の座標と重み
    
    myAnnulus.plotKntSurf()                                                     
    plt.scatter(xPhys, yPhys, s=0.5)
    X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)
    #x_fはガウス積分点の座標と重み
    #→→→ model_pts['X_int'] へ
    
    # Generate the boundary points using Geometry class
    numElemEdge = 80            #要素数
    numGaussEdge = 1            #各要素のガウス点数
    xEdge, yEdge, xNormEdge, yNormEdge, wgtsEdge = myAnnulus.getQuadEdgePts(numElemEdge,
                                                                numGaussEdge, 4)
    trac_x = -model['P'] * xNormEdge
    trac_y = -model['P'] * yNormEdge
    xEdgePts = np.concatenate((xEdge, yEdge, wgtsEdge, trac_x, trac_y), axis=1)
    #物理空間内での点のx座標(physPtX), 物理空間内での点のy座標(physPtY), 重み, トラクションx, トラクションy
    #→→→ model_pts['X_bnd'] へ
    
    model_pts = dict()      
    model_pts['X_int'] = X_f
    model_pts['X_bnd'] = xEdgePts
    
    modelNN = PINN_TC(model_data, xEdgePts, NN_param, X_f)
    
    filename = 'Training_scatter'
    scatterPlot(X_f,xEdgePts,figHeight,figWidth,filename)

    nPred = 40
    withEdges = [1, 1, 1, 1]
    xGrid, yGrid = myAnnulus.getUnifIntPts(nPred, nPred, withEdges)             #パラメタから物理へ？？
    Grid = np.concatenate((xGrid,yGrid),axis=1)
    
    num_train_its = 1000
    numSteps = 4 # Number of refinement steps
    for i in range(numSteps):
        
        # Compute/training
        start_time = time.time()
        modelNN.train(X_f, num_train_its)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        print("Degrees of freedom ", X_f.shape[0])
        # Error estimation
        f_u, f_v = modelNN.predict_f(X_f[:,0:2])
        res_err = np.sqrt(f_u**2 + f_v**2)
                
        numElem = len(vertex)                                                                           #要素数
        errElem = np.zeros(numElem)                                                                     #各要素ごとのerrorを格納するゼロ行列
        for iElem in range(numElem):
            ptIndStart = iElem*numGauss**2                                                              #各要素での積分点のインデックス開始位置
            ptIndEnd = (iElem+1)*numGauss**2                                                            #各要素での積分点のインデックス終了位置
            # Estimate the error in each element by integrating 
            errElem[iElem] = np.sum(res_err[ptIndStart:ptIndEnd]*X_f[ptIndStart:ptIndEnd,2])            #前半:要素内の積分点での残差　後半:各積分点の重み
        
        # Marking the elements for refinement
        N = 10 # N percent interior points with highest error
        ntop = np.int(np.round(numElem*N/100))
        sort_err_ind = np.argsort(-errElem, axis=0) 
        index_ref = np.squeeze(sort_err_ind[0:ntop]) # ←Indices of the elements that are to be refined
        
        #argsortはソートした配列の要素のインデックスを返す axis=0で列ごとに見て要素をソート
        #squeezeで次元を減らす

        # Refine element list
        vertex = refineElemVertex(vertex, index_ref)                                                    #refine後のvertex
        xPhys, yPhys, wgtsPhys = myAnnulus.getElemIntPts(vertex, numGauss)                              #refine後の新しい積分点
        X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)
    
    
        filename = 'Refined_scatter'+ str(i)
        scatterPlot(X_f,xEdgePts,figHeight,figWidth,filename)                                           #プロット
        
        u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(X_f)     #予測
        energy_err, energy_norm = energyError(X_f,sigma_x_pred,sigma_y_pred,model,tau_xy_pred)          #エネルギーノルム、エネルギー誤差
        print("Relative energy error (integration): ", np.sqrt(energy_err/energy_norm))
        
    # Plot results
    # Magnification factors for plotting the deformed shape　拡大係数
    x_fac = 2
    y_fac = 2
    
    # Compute the approximate displacements at plot points
    u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(Grid)
    u_exact, v_exact = getExactDisplacements(xGrid, yGrid, model) # Computing exact displacements 
    oShapeX = np.resize(xGrid, [nPred, nPred])                      #未変形メッシュの各頂点のx座標                     #resizeは配列の変形　np.resize(a,(3,3))
    oShapeY = np.resize(yGrid, [nPred, nPred])                      #未変形メッシュの各頂点のy座標    
    surfaceUx = np.resize(u_pred, [nPred, nPred])                   #予測変形データ
    surfaceUy = np.resize(v_pred, [nPred, nPred])
    surfaceExUx = np.resize(u_exact, [nPred, nPred])                #正確な変形データ
    surfaceExUy = np.resize(v_exact, [nPred, nPred])
    
    defShapeX = oShapeX + surfaceUx * x_fac
    defShapeY = oShapeY + surfaceUy * y_fac
    surfaceErrUx = surfaceExUx - surfaceUx
    surfaceErrUy = surfaceExUy - surfaceUy      

    #変位データのプロット       
    print("Deformation plots")
    filename = 'Deformation'
    plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY, filename)
    
    print("Exact plots")
    filename = 'Exact'
    plotDeformedDisp(surfaceExUx, surfaceExUy, defShapeX, defShapeY, filename)
    
    print("Error plots")
    filename = 'Error'
    plotDeformedDisp(surfaceErrUx, surfaceErrUy, oShapeX, oShapeY, filename)
    
    # Plotting the strain energy densities 
    filename = 'Strain_energy'       
    energyPlot(defShapeX,defShapeY,nPred,energy_pred,filename,figHeight,figWidth)

    # Compute the L2 and energy norm errors using integration
    u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(X_f)             #予測結果
    u_exact, v_exact = getExactDisplacements(X_f[:,0], X_f[:,1], model)                                     #正確な変位データ
    err_l2 = np.sum(((u_exact-u_pred[:,0])**2 + (v_exact-v_pred[:,0])**2)*X_f[:,2])                         #L2ノルムエラー
    norm_l2 = np.sum((u_exact**2 + v_exact**2)*X_f[:,2])                                                    #正確な変位のL2ノルム
    error_u_l2 = np.sqrt(err_l2/norm_l2)                                                                    #エラーの正規化
    print("Relative L2 error (integration): ", error_u_l2)
    
    energy_err, energy_norm = energyError(X_f,sigma_x_pred,sigma_y_pred,model,tau_xy_pred)                  #エネルギーノルムとそのエラー
    print("Relative energy error (integration): ", np.sqrt(energy_err/energy_norm))
    
    os.chdir(originalDir)
