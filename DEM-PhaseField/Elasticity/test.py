"""
File for base geometry class built using the Geomdl class
"""

import numpy as np
from geomdl import NURBS
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


class Geometry2D:
    '''
     Base class for 2D domains
     Input: geomData - dictionary containing the geomety information
     Keys: degree_u, degree_v: polynomial degree in the u and v directions
       ctrlpts_size_u, ctrlpts_size_v: number of control points in u,v directions
       ctrlpts: weighted control points (in a list with 
            ctrlpts_size_u*ctrlpts_size_v rows and 3 columns for x,y,z coordinates)
       weights: correspond weights (list with ctrlpts_size_u*ctrlpts_size_v entries)
       knotvector_u, knotvector_v: knot vectors in the u and v directions
    '''
    def __init__(self, geomData):
        self.surf = NURBS.Surface()
        self.surf.degree_u = geomData['degree_u']
        self.surf.degree_v = geomData['degree_v']
        self.surf.ctrlpts_size_u = geomData['ctrlpts_size_u']
        self.surf.ctrlpts_size_v = geomData['ctrlpts_size_v']
        self.surf.ctrlpts = self.getUnweightedCpts(geomData['ctrlpts'], 
                                             geomData['weights'])
        self.surf.weights = geomData['weights']
        self.surf.knotvector_u = geomData['knotvector_u']
        self.surf.knotvector_v = geomData['knotvector_v']                
        
    def getUnweightedCpts(self, ctrlpts, weights):
        numCtrlPts = np.shape(ctrlpts)[0]   # ctrlptsの行の数を取得している
        PctrlPts = np.zeros_like(ctrlpts)
        for i in range(3):
            for j in range(numCtrlPts):
                PctrlPts[j,i]=ctrlpts[j][i]/weights[j]
        PctrlPts = PctrlPts.tolist()        #リスト化　[1 2 3] → [1, 2, 3]
        return PctrlPts
    #重みなしのコントロールポイント
        
    def mapPoints(self, uPar, vPar):
        '''
        Map points from the parameter domain [0,1]x[0,1] to the quadrilater domain
        Input:  uPar - array containing the u-coordinates in the parameter space　０～１だよ
                vPar - array containing the v-coordinates in the parameter space
                Note: the arrays uPar and vPar must be of the same size
        Output: xPhys - array containing the x-coordinates in the physical space
                yPhys - array containing the y-coordinates in the physical space
        '''        
        gpParamUV = np.array([uPar, vPar])                          #各点におけるuv座標
        evalList = tuple(map(tuple, gpParamUV.transpose()))         #転置してタプルのリストに変更　（各タプルが(u,v)座標を表す）
        res = np.array(self.surf.evaluate_list(evalList))           #評価　　#self.surfはNURBS 137
                
        return res
    
    def getUnifIntPts(self, numPtsU, numPtsV, withEdges):
        '''
        Generate uniformly spaced points inside the domain
        Input: numPtsU, numPtsV - number of points (including edges) in the u and v
                   directions in the parameter space
               withEdges - 1x4 array of zeros or ones specifying whether the boundary points
                           should be included. The boundary order is [bottom, right,
                           top, left] for the unit square.   [0, 0, 0, 0]
        Output: xM, yM - flattened array containing the x and y coordinates of the points

        与えられたパラメータ空間内で一様に配置された点を生成し、それらの点を別の座標系に写像するプロセスを実行する関数
        '''
        #generate points in the x direction on the interval [0,1]
        uEdge = np.linspace(0, 1, numPtsU)
        vEdge = np.linspace(0, 1, numPtsV)
        
        #remove endpoints depending on values of withEdges
        if withEdges[0]==0:             #下端の境界点は含まれない
            vEdge = vEdge[1:]           #スライシング　v方向の最初の0を取り除く
        if withEdges[1]==0:             #右端の境界点は含まれない
            uEdge = uEdge[:-1]          #スライシング　u方向の最後の1を取り除く
        if withEdges[2]==0:
            vEdge = vEdge[:-1]
        if withEdges[3]==0:
            uEdge = uEdge[1:]
            
        #create meshgrid
        uPar, vPar = np.meshgrid(uEdge, vEdge)        
                        
        uPar = uPar.flatten()
        vPar = vPar.flatten()     
        #map points
        res = self.mapPoints(uPar.T, vPar.T)        #物理空間でのxy座標（スライス済み）
        
        xPhys = res[:, 0:1]     #res配列の1列目のデータを抽出
        yPhys = res[:, 1:2]     #res配列の2列目のデータを抽出
        
        return xPhys, yPhys
    
    def genElemList(self, numElemU, numElemV):
        '''
        Generate quadrature points inside the domain for an initial (uniform)
        subdivision mesh
        Input: numElemU, numElemV - number of subdivisions in the u and v
                   directions in the parameter space               
        Output: vertex - arrays containing the element vertices

        初期の一様な分割メッシュの要素領域内に、特定の数の要素（四角形）を生成する関数
        '''
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
                                        
        return vertex
    
    def getElemIntPts(self, vertex, numGauss):
        '''
        Generate quadrature points inside the domain
        Input: vertex - contains the vertices of the elements the refined elements
               numGauss - number of Gauss quadrature points for each subdivision
        Output: xPhys, yPhys, wgtPhy - arrays containing the x and y coordinates
                                    of the points and the corresponding weights
        
        指定された要素領域内において、ガウス積分点を生成する
        '''
        #allocate quadPts array
        numElems = vertex.shape[0]                              #vetexの行数=要素の数
        quadPts = np.zeros((numElems*numGauss**2, 3))           #各行は1つの積分点の座標と重みを含む
        
        # Get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)      #gpは点の位置, gwは重み
        
        #get the Gauss weights on the reference element [-1, 1]x[-1,1]
        gpWeightU, gpWeightV = np.meshgrid(gw, gw)              #重みを格納
        gpWeightUV = np.array(gpWeightU.flatten()*gpWeightV.flatten())      #ガウス積分点ごとの重み
               
        #generate points for each element
        indexPt = 0
        for iElem in range(numElems):       #各要素についてループ
            
            uMin = vertex[iElem,0]
            uMax = vertex[iElem,2]
            vMin = vertex[iElem,1]
            vMax = vertex[iElem,3]
            gpParamU = (uMax-uMin)/2*gp+(uMax+uMin)/2                                   #各ガウス積分点が参照要素のパラメータ空間でどの位置にあるか示す
            gpParamV = (vMax-vMin)/2*gp+(vMax+vMin)/2
            gpParamUg, gpParamVg = np.meshgrid(gpParamU, gpParamV)
            gpParamUV = np.array([gpParamUg.flatten(), gpParamVg.flatten()])            #ガウス積分点の位置　2次元のパラメータ空間で
            #Jacobian of the transformation from the reference element [-1,1]
            scaleFac = (uMax-uMin)*(vMax-vMin)/4                                        #座標変換において、異なる座標系の間で距離、面積、体積などの尺度を変更するために使用される係数
                
                #map the points to the physical space
            for iPt in range(numGauss**2):
                curPtU = gpParamUV[0, iPt]                                              #ガウス積分点のパラメータ空間内での位置
                curPtV = gpParamUV[1, iPt]
                derivMat = self.surf.derivatives(curPtU, curPtV, order=1)               #ガウス積分点における1階導関数の計算
                physPtX = derivMat[0][0][0]                                             #物理空間のおけるx座標
                physPtY = derivMat[0][0][1]
                derivU = derivMat[1][0][0:2]                                            #u方向の導関数成分を抽出
                derivV = derivMat[0][1][0:2]
                JacobMat = np.array([derivU,derivV])                                    #ヤコビアン行列の生成
                detJac = np.linalg.det(JacobMat)                                        #ヤコビアン行列を求める　座標変換の勾配を表す
                quadPts[indexPt, 0] = physPtX
                quadPts[indexPt, 1] = physPtY
                quadPts[indexPt, 2] = scaleFac * detJac * gpWeightUV[iPt]               #積分点の物理空間内での重み
                indexPt = indexPt + 1                                                   #各要素で+1
        
        xPhys = quadPts[:, 0:1]
        yPhys = quadPts[:, 1:2]
        wgtPhys = quadPts[:, 2:3]
        
        return xPhys, yPhys, wgtPhys                #ガウス積分点の位置と重みの情報
    
    def getUnweightedCpts2d(self, ctrlpts2d, weights):
        '''
        制御点（Control Points）の2次元配列とそれに対応する重みを使用して、重みを適用した制御点の座標を計算し、結果を返す
        '''
        numCtrlPtsU = np.shape(ctrlpts2d)[0]        #制御点の数
        numCtrlPtsV = np.shape(ctrlpts2d)[1]
        PctrlPts = np.zeros([numCtrlPtsU,numCtrlPtsV,3])            #重みを適用した制御点の座標を格納するための3次元配列の初期化
        counter = 0    
        for j in range(numCtrlPtsU):
            for k in range(numCtrlPtsV):
                for i in range(3):
                    PctrlPts[j,k,i]=ctrlpts2d[j][k][i]/weights[counter]         #制御点の座標を重みで割って新しい座標を計算
                counter = counter + 1
        PctrlPts = PctrlPts.tolist()            #NumPyの多次元配列をリストに変換
        return PctrlPts
    
    
    def plotSurf(self):
        '''
        NURBS（Non-Uniform Rational B-Spline）またはB-Spline曲面を可視化し、2D平面上に制御点と曲面を描画する
        '''
        #plots the NURBS/B-Spline surface and the control points in 2D
        fig, ax = plt.subplots()
        patches = []
            
        #get the number of points in the u and v directions
        numPtsU = np.int(1/self.surf.delta[0])
        numPtsV = np.int(1/self.surf.delta[1])
        print(numPtsU, numPtsV)

        #曲面を可視化するための要素（Polygon）を生成する
        for j in range(numPtsV-1):                        
            for i in range(numPtsU-1):
                #get the index of point in the lower left corner of the visualization element
                indexPtSW = j*numPtsU + i                                       #要素の左下の角の制御点のインデックス
                indexPtSE = indexPtSW + 1                                           #要素の右下の角の制御点のインデックス
                indexPtNE = indexPtSW + numPtsU + 1                                 #要素の右上の角の制御点のインデックス
                indexPtNW = indexPtSW + numPtsU                                     #要素の左上の角の制御点のインデックス
                #上記で計算した制御点のインデックスを使用して、曲面の評価点から対応する座標を取得
                #np.array(self.surf.evalpts) は曲面の評価点を2D NumPy配列として取得し、各角に対応する座標を選択
                #XYPts には、4つの角に対応する座標が格納される
                print(j)
                print(indexPtSW, indexPtSE, indexPtNE, indexPtNW)
                XYPts = np.array(self.surf.evalpts)[[indexPtSW, indexPtSE,                      
                                indexPtNE, indexPtNW],0:2]
                #for xx in range(400):
                    #print(str(self.surf.evalpts[xx][0:2])[1:-1])
                poly = mpatches.Polygon(XYPts)  #, ec="none")　ポリゴンオブジェクトを生成
                patches.append(poly)            #ポリゴンオブジェクトをpatchesリストに追加
                
                
        collection = PatchCollection(patches, color="lightgreen", cmap=plt.cm.hsv, alpha=1)                         #patchリスト内の要素をまとめてPatchCollectionオブジェクトに配置
        ax.add_collection(collection)                                                                               #PatchCollectionを座標軸axに追加
        
        numCtrlPtsU = self.surf._control_points_size[0]         #uv方向の制御点の数を取得
        numCtrlPtsV = self.surf._control_points_size[1]
        ctrlpts = self.getUnweightedCpts2d(self.surf.ctrlpts2d, self.surf.weights)                                  #制御点の座標に重みを適用せず制御点の座標を返す
        #plot the horizontal lines
        for j in range(numCtrlPtsU):
            plt.plot(np.array(ctrlpts)[j,:,0],np.array(ctrlpts)[j,:,1],ls='--',color='black')                       #j行目
        #plot the vertical lines
        for i in range(numCtrlPtsV):
            plt.plot(np.array(ctrlpts)[:,i,0],np.array(ctrlpts)[:,i,1],ls='--',color='black')                       #i列目
        #plot the control points
        plt.scatter(np.array(self.surf.ctrlpts)[:,0],np.array(self.surf.ctrlpts)[:,1],color='red',zorder=10)        #制御点のxy座標をプロット
        plt.axis('equal')       #x軸とy軸のスケールを等しくし、プロットを等比例に表示する設定を行います。これにより、描画された制御ポリゴンが適切に表示されます。
        
    def plotKntSurf(self):
        '''
        NURBS/B-Spline曲面とそれに関連するノットライン（ノットベクトルから生成される線）を2Dで描画する
        '''
        #plots the NURBS/B-Spline surface and the knot lines in 2D
        fig, ax = plt.subplots()
        patches = []
        
        #get the number of points in the u and v directions
        self.surf.delta = 0.02
        self.surf.evaluate()                                            #曲面を評価します。これにより、曲面上の評価点が計算されます。
        numPtsU = np.int(1/self.surf.delta[0])                          #描画領域内の点の数を計算
        numPtsV = np.int(1/self.surf.delta[1])
        
        for j in range(numPtsV):
            for i in range(numPtsU):
                #get the index of point in the lower left corner of the visualization element
                indexPtSW = j*(numPtsU+1) + i
                indexPtSE = indexPtSW + 1
                indexPtNE = indexPtSW + numPtsU + 2
                indexPtNW = indexPtSW + numPtsU + 1
                XYPts = np.array(self.surf.evalpts)[[indexPtSW, indexPtSE, indexPtNE, indexPtNW],0:2]       #インデックスから座標を取得　xy座標を持つ2dnumpy配列
                poly = mpatches.Polygon(XYPts)
                patches.append(poly)            #ポリゴン化してリストに追加
                
        collection = PatchCollection(patches, color="lightgreen", cmap=plt.cm.hsv, alpha=1)         #PatchCollectionは曲面を構成する要素の可視化を担当
        ax.add_collection(collection)                                                               #PatchCollectionをaxに追加して曲面の要素を描画領域に表示
        
        #plot the horizontal knot lines
        for j in np.unique(self.surf.knotvector_u):
            vVal = np.linspace(0, 1, numPtsV)               #0から1までの範囲で等間隔に値を生成し、numPtsV 個の評価点のv座標を表す
            uVal = np.ones(numPtsV)*j                       # j を持つu座標の値を生成
            uvVal = np.array([uVal, vVal])
            
            evalList=tuple(map(tuple, uvVal.transpose()))                           #uvValの行をタプルに変換し、評価リストを作成
            res=np.array(self.surf.evaluate_list(evalList))                         #曲面を評価し、評価点の座標情報を返す
            plt.plot(res[:,0],res[:,1], ls='-', linewidth=1, color='black')         #水平ノットラインを描画
            
        #plot the vertical lines
        for i in np.unique(self.surf.knotvector_v):
            uVal = np.linspace(0, 1, numPtsU)
            vVal = np.ones(numPtsU)*i    
            uvVal = np.array([uVal, vVal])
            
            evalList=tuple(map(tuple, uvVal.transpose()))
            res=np.array(self.surf.evaluate_list(evalList))        
            plt.plot(res[:,0],res[:,1], ls='-', linewidth=1, color='black')         #垂直ノットラインを描画
       
        plt.axis('equal')        
    
    def getQuadEdgePts(self, numElem, numGauss, orient):
        '''
        Generate points on the boundary edge given by orient
        Input: numElem - number of number of subdivisions (in the v direction)
               numGauss - number of Gauss points per subdivision　　　各方向のガウスポイント数
               orient - edge orientation in parameter space: 1 is down (v=0), 
                        2 is left (u=1), 3 is top (v=1), 4 is right (u=0)  ２と4逆じゃない？？？
        Output: xBnd, yBnd, wgtBnd - coordinates of the boundary in the physical
                                     space and the corresponding weights
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
                #allocate quadPts array
        quadPts = np.zeros((numElem*numGauss, 5))
        
        #get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)         #gpは座標、gwは重み         
        
        #generate the knots on the interval [0,1]
        edgePar = np.linspace(0, 1, numElem+1)            
                        
        #generate points for each element
        indexPt = 0
        for iE in range(numElem):            #ある方向の全ての要素についてループ    
                edgeMin = edgePar[iE]        
                edgeMax = edgePar[iE+1]      
                if orient==1:
                    gpParamU = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                    gpParamV = np.zeros_like(gp)                    
                elif orient==2:
                    gpParamU = np.ones_like(gp)
                    gpParamV = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                elif orient==3:
                    gpParamU = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                    gpParamV = np.ones_like(gp)   
                elif orient==4:
                    gpParamU = np.zeros_like(gp)
                    gpParamV = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                else:
                    raise Exception('Wrong orientation given')
                        
                gpParamUV = np.array([gpParamU.flatten(), gpParamV.flatten()])      #ガウス点の座標
                
                #Jacobian of the transformation from the reference element [-1,1]
                scaleFac = (edgeMax-edgeMin)/2
                
                #map the points to the physical space
                for iPt in range(numGauss):
                    curPtU = gpParamUV[0, iPt]              #iPt番目のx座標
                    curPtV = gpParamUV[1, iPt]              #iPt番目のy座標
                    derivMat = self.surf.derivatives(curPtU, curPtV, order=1)       #そのガウス点での1階導関数
                    physPtX = derivMat[0][0][0]             #物理空間におけるx座標
                    physPtY = derivMat[0][0][1]             
                    derivU = derivMat[1][0][0:2]            #u方向の導関数成分
                    derivV = derivMat[0][1][0:2]
                    JacobMat = np.array([derivU,derivV])    #ヤコビアン行列の生成
                    if orient==1:                           #normX と normY は、法線ベクトルの x 成分と y 成分                    
                        normX = JacobMat[0,1]
                        normY = -JacobMat[0,0]
                    elif orient==2:
                        normX = JacobMat[1,1]
                        normY = -JacobMat[1,0]
                    elif orient==3:
                        normX = -JacobMat[0,1]
                        normY = JacobMat[0,0]
                    elif orient==4:
                        normX = -JacobMat[1,1]
                        normY = JacobMat[1,0]
                    else:
                        raise Exception('Wrong orientation given')
                        
                    JacobEdge = np.sqrt(normX**2+normY**2)                      #法線ベクトルの大きさ
                    normX = normX/JacobEdge                                     #法線ベクトルのx成分
                    normY = normY/JacobEdge

                    '''
                    各ガウスポイントに関する情報を格納
                    列 0: 物理空間内での点の x 座標 (physPtX)
                    列 1: 物理空間内での点の y 座標 (physPtY)
                    列 2: 法線ベクトルの x 成分 (normX)
                    列 3: 法線ベクトルの y 成分 (normY)
                    列 4: 重み (scaleFac * JacobEdge * gw[iPt])
                    '''
        
                    quadPts[indexPt, 0] = physPtX
                    quadPts[indexPt, 1] = physPtY
                    quadPts[indexPt, 2] = normX
                    quadPts[indexPt, 3] = normY
                    quadPts[indexPt, 4] = scaleFac * JacobEdge * gw[iPt]
                    indexPt = indexPt + 1
                            
        xPhys = quadPts[:, 0:1]
        yPhys = quadPts[:, 1:2]
        xNorm = quadPts[:, 2:3]
        yNorm = quadPts[:, 3:4]
        wgtPhys = quadPts[:, 4:5]        
        
        return xPhys, yPhys, xNorm, yNorm, wgtPhys
    
    '''
    この部分は関数 getQuadEdgePts の出力を設定しています。具体的には、以下の情報を返しています：

    xPhys: 物理空間内の点の x 座標の配列。
    yPhys: 物理空間内の点の y 座標の配列。
    xNorm: 物理空間内の各点の法線ベクトルの x 成分の配列。
    yNorm: 物理空間内の各点の法線ベクトルの y 成分の配列。
    wgtPhys: 各点に対する重みの配列。
    つまり、この関数は物理空間内での点の座標、法線ベクトル、および重みを返します。これらの情報は、曲面の特性を解析したり可視化したりするために使用できます。
    ''' 

geomData = dict()

radInt = 1
radExt = 4

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

my_geometry = Geometry2D(geomData)
my_geometry.plotSurf()