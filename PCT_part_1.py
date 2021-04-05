#
#
#

import pandas as pd 
import numpy as np 


class PCT(object):

    def __init__(self, raw, d, n, m):

        self.raw = raw   # import raw data 
        self.d = d       # chose R or L
        self.n = n       # set up numbers of Markerset
        self.m = m
        
    def get_time(self):
        '''
        time series
        '''
        global time

        time = self.raw['Unnamed: 1_level_0'] 
        time.columns = [['Time'],['time']]

        return time

    def get_G(self):
        '''
        Global frame
        '''

        global G
        global G_columns

        G = pd.DataFrame()

        if self.d == 'R':
            
            for g in ['MarkerSet:R{}'.format(i+1) for i in range(self.n)]:
                
                G = pd.concat((G,self.raw[g]), axis=1)

        elif self.d == 'L':
            
            for g in ['MarkerSet:L{}'.format(i+1) for i in range(self.n)]:
                
                G = pd.concat((G,self.raw[g]), axis=1)
        
        G_columns = [['G{}'.format(i+1) for i in range(self.n)],['X','Y','Z']]
        G.columns = pd.MultiIndex.from_product(G_columns)

        return G

    def get_C(self):
        '''
        center of mass
        '''

        global C

        sum_m = np.sum(self.m)
        Gxm = []
        Gym = []
        Gzm = []

        for i, j in zip(G_columns[0], range(len(self.m))):
            Gxm.append(G[i]['X'].values.reshape(-1, 1) * self.m[j])
            Gym.append(G[i]['Y'].values.reshape(-1, 1) * self.m[j])
            Gzm.append(G[i]['Z'].values.reshape(-1, 1) * self.m[j])

        sum_x = np.sum((Gxm), axis=0) / sum_m
        sum_y = np.sum((Gym), axis=0) / sum_m
        sum_z = np.sum((Gzm), axis=0) / sum_m

        C = pd.concat((pd.DataFrame(sum_x), pd.DataFrame(sum_y), pd.DataFrame(sum_z)), axis=1)
        C.columns = ['X', 'Y', 'Z']

        return C

    def get_P(self):
        '''
        vector P position
        '''

        global P
        global P_columns

        P = pd.DataFrame()

        for g in G_columns[0]:

            p = G[g] - C  

            P = pd.concat((P,p), axis=1)

        P_columns = [ ['P{}'.format(i+1) for i in range(self.n)],['X','Y','Z'] ]
        P.columns = pd.MultiIndex.from_product(P_columns)

        return P
        
    def get_I(self):
        '''
        Interia Tensor
        '''
        
        global I 
        global I_columns
        
        i_xx = pd.DataFrame( np.sum( (np.square( [ P[p]['Y'] for p in P_columns[0] ] ) + np.square( [ P[p]['Z'] for p in P_columns[0] ] ) ), axis=0 ) )
        i_yy = pd.DataFrame( np.sum( (np.square( [ P[p]['X'] for p in P_columns[0] ] ) + np.square( [ P[p]['Z'] for p in P_columns[0] ] ) ), axis=0 ) )
        i_zz = pd.DataFrame( np.sum( (np.square( [ P[p]['X'] for p in P_columns[0] ] ) + np.square( [ P[p]['Y'] for p in P_columns[0] ] ) ), axis=0 ) )
        i_xy = pd.DataFrame( np.sum( ( [ P[p]['X']*P[p]['Y'] for p in P_columns[0] ] ), axis=0) )
        i_xz = pd.DataFrame( np.sum( ( [ P[p]['X']*P[p]['Z'] for p in P_columns[0] ] ), axis=0) )
        i_yz = pd.DataFrame( np.sum( ( [ P[p]['Y']*P[p]['Z'] for p in P_columns[0] ] ), axis=0) )
        
        I = pd.concat((i_xx,-i_xy,-i_xz,-i_xy,i_yy,-i_yz,-i_xz,-i_yz,i_zz),axis=1 )
        
        I_columns = ['i_xx','-i_xy','-i_xz','-i_xy','i_yy','-i_yz','-i_xz','-i_yz','i_zz']
        I.columns = I_columns
        
        return I 
    
    def get_Eigen(self):
        '''
        eigenvalue
        eigenvector
        '''

        global E_val
        global E_vec

        E_val = pd.DataFrame()
        E_vec = pd.DataFrame()

        for count in range(len(I)):

            eigval, eigvec = np.linalg.eig(I.loc[count,:].values.reshape(3,3))

            E_val = E_val.append(pd.DataFrame(eigval.reshape(1,-1)))
            E_vec = E_vec.append(pd.DataFrame(eigvec.reshape(1,-1)))
        
        E_val.index = [ count for count in range(len(E_val))]
        E_val.columns = ['lambda_1', 'lambda2', 'lambda_3']

        E_vec_columns = [ ['E1','E2','E3'],['X','Y','Z'] ]
        E_vec.index = [count for count in range(len(E_vec))]
        E_vec.columns = pd.MultiIndex.from_product(E_vec_columns)

        return E_val, E_vec
    
    def get_L(self):
        '''
        Local frame
        '''

        L = pd.DataFrame()

        for g in G_columns[0]:

            for count in range(len(E_vec)):

                R = np.matrix(E_vec.iloc[count,:].values.reshape(3,3))

                l = R.T * ( G[g].iloc[count,:].values.reshape(-1,1) - C.iloc[count,:].values.reshape(-1,1) )

                L = pd.concat( (L,pd.DataFrame(l.reshape(1,-1))), axis=0)
        
        L_index = [ ['L{}'.format(i+1) for i in range(self.n) ], [i for i in range(len(E_vec))]  ]
        L_columns = [ ['L{}'.format(i+1) for i in range(self.n)],['X','Y','Z'] ]

        L.index = pd.MultiIndex.from_product(L_index)
        L.columns = L_columns[1]

        _l = pd.DataFrame()

        for l in L_index[0]:

            _l = pd.concat( (_l,L.loc[l,:]), axis=1 )

        _l.columns = pd.MultiIndex.from_product(L_columns)

        L = pd.concat( (time,_l), axis=1 )

        return L