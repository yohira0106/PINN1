# Class file for Physics Informed Neural Network

#import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow as tf 
import numpy as np
from geomdl import NURBS

surf = NURBS.Surface()

numElemU = 20       #u方向要素数
numElemV = 20


vertex = np.zeros((numElemU*numElemV, 4))       #各要素には4つの頂点が含まれる
                        
#generate the knots on the interval [0,1]
uEdge = np.linspace(0, 1,  numElemU+1)
vEdge = np.linspace(0, 1, numElemV+1)            
#totalGaussPts = numGauss**2
#create meshgrid
uPar, vPar = np.meshgrid(uEdge, vEdge)              
counterElem = 0                                 #要素をカウントするための変数
#generate points for each element

for iV in range(numElemV):
    for iU in range(numElemU):
        uMin = uPar[iV, iU]
        uMax = uPar[iV, iU+1]
        vMin = vPar[iV, iU]
        vMax = vPar[iV+1, iU]                
        vertex[counterElem, 0] = uMin
        vertex[counterElem, 1] = vMin
        vertex[counterElem, 2] = uMax
        vertex[counterElem, 3] = vMax
        counterElem = counterElem + 1
                                        

numGauss = 2

gp, gw = np.polynomial.legendre.leggauss(numGauss) 
print('gp, gw:', gp, gw)

quadPts = np.zeros((numElemU*numGauss, 2))

gpWeightU, gpWeightV = np.meshgrid(gw, gw)              #重みを格納
gpWeightUV = np.array(gpWeightU.flatten()*gpWeightV.flatten()) 

uMin = vertex[0,0]
uMax = vertex[0,2]
vMin = vertex[0,1]
vMax = vertex[0,3]
print('uMin:', uMin)
print('uMax:', uMax)
print('vMin:', vMin)
print('vMax:', vMax)

gpParamU = (uMax-uMin)/2*gp+(uMax+uMin)/2         
print('gpParamU:', gpParamU)
gpParamV = (vMax-vMin)/2*gp+(vMax+vMin)/2
print('gpParamV:', gpParamV)
gpParamUg, gpParamVg = np.meshgrid(gpParamU, gpParamV)
print('gpParamUg:', gpParamUg)
print('gpParamVg:', gpParamVg)
gpParamUV = np.array([gpParamUg.flatten(), gpParamVg.flatten()]) 
print('gpParamUV:', gpParamUV)

#_______________________________

radInt = 1.0   #内径
radExt = 4.0

geomData = dict()
geomData['degree_u'] = 1
geomData['degree_v'] = 2
        
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

surf.degree_u = geomData['degree_u']
surf.degree_v = geomData['degree_v']
ctrlpts_size_u = geomData['ctrlpts_size_u']
surf.ctrlpts_size_v = geomData['ctrlpts_size_v']

def getUnweightedCpts(ctrlpts, weights):
    numCtrlPts = np.shape(ctrlpts)[0]   # ctrlptsの行の数を取得している
    PctrlPts = np.zeros_like(ctrlpts)
    for i in range(3):
        for j in range(numCtrlPts):
            PctrlPts[j,i]=ctrlpts[j][i]/weights[j]
    PctrlPts = PctrlPts.tolist()        #リスト化　[1 2 3] → [1, 2, 3]
    return PctrlPts


surf.weights = geomData['weights']
surf.knotvector_u = geomData['knotvector_u']
surf.knotvector_v = geomData['knotvector_v']

scaleFac = (uMax-uMin)*(vMax-vMin)/4 

for iPt in range(numGauss**2):
    curPtU = gpParamUV[0, iPt]                                              #ガウス積分点のパラメータ空間内での位置
    curPtV = gpParamUV[1, iPt]
    derivMat = surf.derivatives(curPtU, curPtV, order=1)               #ガウス積分点における1階導関数の計算
    physPtX = derivMat[0][0][0]                                             #物理空間のおけるx座標
    physPtY = derivMat[0][0][1]
    derivU = derivMat[1][0][0:2]                                            #u方向の導関数成分を抽出
    derivV = derivMat[0][1][0:2]
    JacobMat = np.array([derivU,derivV])                                    #ヤコビアン行列の生成
    detJac = np.linalg.det(JacobMat)                                        #ヤコビアン行列を求める　座標変換の勾配を表す
    quadPts[0, 0] = physPtX
    quadPts[0, 1] = physPtY
    quadPts[0, 2] = scaleFac * detJac * gpWeightUV[iPt] 