import os
import numpy as np
import matplotlib.pyplot as plt

def createFolder(folder_name):
    try:    #例外が発生するかもしれないが、実行したい処理
        if not os.path.exists(folder_name):     #パスが存在するとTrueを返す
            os.makedirs(folder_name)            #ディレクトリを作成
    except OSError:     #例外発生時に行う処理
        print ('Error: Creating folder. ' +  folder_name)

def cart2pol(x, y):
    rho = np.sqrt(np.array(x)**2 + np.array(y)**2)
    phi = np.arctan2(y, x)      #arctan(y/x)を返す arctan()は引数が1つ
    return(rho, phi)

def getExactDisplacements(x,y,model):
    r, th = cart2pol(x,y)       #rとΘ
    #論文式36
    u_r = model['radInt']**2*model['P']*r/(model['E']* \
               (model['radExt']**2-model['radInt']**2)) * \
               (1-model['nu']+(model['radExt']/r)**2 * (1 + model['nu']))
    u_exact = u_r*np.cos(th)               
    v_exact = u_r*np.sin(th)
               
    return u_exact, v_exact

def getExactStresses(x, y, model):
    sigma_xx = np.zeros_like(x) #要素を0で初期化する
    sigma_yy = np.zeros_like(x)
    sigma_xy = np.zeros_like(x)
    numPts = len(x)     #積分点の数
    #論文式35
    for i in range(numPts):
        r, th = cart2pol(x[i],y[i])     #x,yからrとθを求める
        sigma_rr = model['radInt']**2*model['P']/(model['radExt']**2-model['radInt']**2) \
                    *(1-model['radExt']**2/r**2)
        sigma_tt = model['radInt']**2*model['P']/(model['radExt']**2-model['radInt']**2) \
                    *(1+model['radExt']**2/r**2)
        sigma_rt = 0
        
        A = np.array([[np.cos(th)**2, np.sin(th)**2, 2*np.sin(th)*np.cos(th)],
                       [np.sin(th)**2, np.cos(th)**2, -2*np.sin(th)*np.cos(th)],
                       [-np.sin(th)*np.cos(th), np.sin(th)*np.cos(th), np.cos(th)**2-np.sin(th)**2]])

        stress_vec = np.linalg.solve(A, np.array([sigma_rr, sigma_tt, sigma_rt]))       #A * stress_vec = np.array[極座標] の解
        sigma_xx[i] = stress_vec[0]
        sigma_yy[i] = stress_vec[1]
        sigma_xy[i] = stress_vec[2]
    return sigma_xx, sigma_yy, sigma_xy
    #直交座標の応力分布を返す

def scatterPlot(X_f,X_bnd,figHeight,figWidth,filename):

    plt.figure(figsize=(figWidth,figHeight))    #図のサイズを指定
    plt.scatter(X_f[:,0], X_f[:,1],s=1.0)   #すべての行の0列目, 全ての行の1列目, プロットのサイズ
    plt.scatter(X_bnd[:,0], X_bnd[:,1], s=1.0, c='red', zorder=10 ) #zorderが大きいほうが上にプロットされる
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12) #メモリ
    plt.yticks(fontsize=12)
    plt.tight_layout()      #サブプロット間の正しい感覚を自動的に維持する
    plt.savefig(filename +'.pdf',dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    #dpi解像度, transparent背景の透明化, bbox_inchesをtightにすると余白を削除する
    plt.show()
    plt.close()
    
def refineElemVertex(vertex, refList):
    #refines the elements in vertex with indices given by refList by splitting 
    #each element into 4 subdivisions
    #Input: vertex - array of vertices in format [umin, vmin, umax, vmax]
    #       refList - list of element indices to be refined
    #Output: newVertex - refined list of vertices
    
    numRef = len(refList)                                           #refineする要素の数
    newVertex = np.zeros((4*numRef,4))                              #4要素を含むかたまりが4*numRef個（イメージ） 4なのは要素を4つに分解するから
    for i in range(numRef):        
        elemIndex = refList[i]
        uMin = vertex[elemIndex, 0]
        vMin = vertex[elemIndex, 1]
        uMax = vertex[elemIndex, 2]
        vMax = vertex[elemIndex, 3]
        uMid = (uMin+uMax)/2
        vMid = (vMin+vMax)/2
        newVertex[4*i, :] = [uMin, vMin, uMid, vMid]                #左下
        newVertex[4*i+1, :] = [uMid, vMin, uMax, vMid]              #右下
        newVertex[4*i+2, :] = [uMin, vMid, uMid, vMax]              #左上
        newVertex[4*i+3, :] = [uMid, vMid, uMax, vMax]              #右上
    vertex = np.delete(vertex, refList, axis=0)
    newVertex = np.concatenate((vertex, newVertex))                 #結合
    
    return newVertex

def energyPlot(defShapeX,defShapeY,nPred,energy_pred,filename,figHeight,figWidth):    
           
    sEnergy = np.resize(energy_pred, [nPred, nPred])                        #
    plt.figure(figsize=(figWidth, figHeight))
    plt.contourf(defShapeX, defShapeY, sEnergy, 255, cmap=plt.cm.jet)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14) 
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename +".png",dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()

    #エネルギーのプロット
    
def energyError(X_f,sigma_x_pred,sigma_y_pred,model,tau_xy_pred):
    
    sigma_xx_exact, sigma_yy_exact, sigma_xy_exact = getExactStresses(X_f[:,0], X_f[:,1], model)    #全ての行の0列目、全ての行の1列目
    sigma_xx_err = sigma_xx_exact - sigma_x_pred[:,0]
    sigma_yy_err = sigma_yy_exact - sigma_y_pred[:,0]
    sigma_xy_err = sigma_xy_exact - tau_xy_pred[:,0]
    
    energy_err = 0
    energy_norm = 0
    numPts = X_f.shape[0]   #.shape[0]で行の数を取り出している　それが積分点の数
    
    C_mat = np.zeros((3,3))
    C_mat[0,0] = model['E']/(1-model['nu']**2)
    C_mat[1,1] = model['E']/(1-model['nu']**2)
    C_mat[0,1] = model['E']*model['nu']/(1-model['nu']**2)
    C_mat[1,0] = model['E']*model['nu']/(1-model['nu']**2)
    C_mat[2,2] = model['E']/(2*(1+model['nu']))
    C_inv = np.linalg.inv(C_mat)
    for i in range(numPts):
        err_pt = np.array([sigma_xx_err[i],sigma_yy_err[i],sigma_xy_err[i]])
        norm_pt = np.array([sigma_xx_exact[i],sigma_yy_exact[i],sigma_xy_exact[i]])
        energy_err = energy_err + err_pt@C_inv@err_pt.T*X_f[i,2]
        energy_norm = energy_norm + norm_pt@C_inv@norm_pt.T*X_f[i,2]
        
    return energy_err, energy_norm
#エネルギー誤差とエネルギーノルムを返す

def plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY, name):
#defShapeが変形後の形状で、SurfaceUが変位かな
        
    filename = name + 'X'
    plt.figure()
    plt.contourf(defShapeX, defShapeY, surfaceUx, 255, cmap=plt.cm.jet)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14) 
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename +".png",dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    filename = name + 'Y'
    plt.figure()
    plt.contourf(defShapeX, defShapeY, surfaceUy, 255, cmap=plt.cm.jet)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14) 
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename +".png",dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
#変位をプロットする