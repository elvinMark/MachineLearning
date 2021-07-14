import numpy as np

class LinearRegression:
    def __init__(self):
        pass
    
    def fit(self,x_data,y_data):
        N, self.nfeatures = x_data.shape
        M, self.ntargets = y_data.shape

        if N!=M:
            assert("input and target dimension does not match")

        x_data = x_data.reshape((N,self.nfeatures,1))
        y_data = y_data.reshape((N,self.ntargets,1))
        A = np.zeros((self.nfeatures + 1,self.nfeatures + self.ntargets))
        b = np.zeros((self.ntargets,self.nfeatures + self.ntargets))

        A[:self.nfeatures,:self.nfeatures] = np.sum(np.matmul(x_data,x_data.transpose((0,2,1))),axis=0)
        A[:self.nfeatures,self.nfeatures:] = np.sum(np.matmul(x_data,np.ones((N,1,self.ntargets))),axis=0)
        A[self.nfeatures:,:self.nfeatures] = np.sum(x_data.transpose((0,2,1)),axis=0)
        A[self.nfeatures:,self.nfeatures:] = N*np.ones((1,self.ntargets))

        b[:,:self.nfeatures] = np.sum(np.matmul(y_data,x_data.transpose((0,2,1))),axis=0)
        b[:,self.nfeatures:] = np.sum(np.matmul(y_data,np.ones((N,1,self.ntargets))),axis=0)

        sol = b @ np.linalg.pinv(A)
        self.coef = sol[:,:self.nfeatures]
        self.intercept = sol[:,self.nfeatures]
        
    def predict(self,x_data):
        return (self.coef @ x_data.T).T + self.intercept

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components

    def fit(self,x_data):
        self.cov = np.cov(x_data.T)
        self.eigvals,self.eigvecs = np.linalg.eig(self.cov)
        self.transform_matrix = self.eigvecs[:,np.argsort(self.eigvals)[:-self.n_components - 1:-1]]
        
    def transform(self,x_data):
        return x_data @ self.transform_matrix

    def fit_transform(self,x_data):
        self.fit(x_data)
        return self.transform(x_data)

class KMeans:
    def __init__(self,n_clusters,max_iter=200):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self,x_data):
        self.means = list(x_data[:self.n_clusters])

        for i in range(self.max_iter):
            distances = []
            for u in self.means:
                d = x_data - u
                distances.append(np.sum(d*d,axis=1))
            distances = np.array(distances)
            idxs = np.argmin(distances,axis=0)

            self.means = []
            for j in range(self.n_clusters):
                self.means.append(np.mean(x_data[idxs==j],axis=0))

    def predict(self,x_data):
        distances = []
        for u in self.means:
            d = x_data - u
            distances.append(np.sum(d*d,axis=1))
        distances = np.array(distances)
        idxs = np.argmin(distances,axis=0)
        return idxs

    def fit_predict(self,x_data):
        self.fit(x_data)
        return self.predict(x_data)
