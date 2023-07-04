import numpy as np
import pandas as pd

from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import GraphicalLasso

from sklearn import covariance
from sklearn import cluster , manifold

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pylab as pl
import matplotlib.pyplot as plt


class Market:
    
    def __init__(self, data):
        self.data = data.copy()
        self.preprocessing(data)
    
    def preprocessing(self, data):
        self.X = (data - data.mean(axis = 0)) / data.std().copy()
        col = self.X.isna().sum().sort_values()[self.X.isna().sum().sort_values() > 0].index
        self.X.drop(columns = col, inplace = True)
        self.liste = list(self.X.columns)
        
    def get_covariance(self, X):
        edge_model = covariance.GraphicalLassoCV(max_iter = 10000)
        edge_model.fit(X)
        return edge_model
        
    def clustering(self, X):
        affinity = cluster.AffinityPropagation()
        affinity.fit(X)
        
        labels = affinity.labels_
        n_labels = labels.max()+1
        names = np.array(self.liste)
        cluster_ = {}
        for i in range(n_labels):
            member = labels == i
            cluster_['cluster '+str(i)] = names[member]
        
        return cluster_, labels
    
    def dimension_reduction(self, X, n_comp = 2):
        reduct = manifold.MDS(n_components = n_comp, random_state=0)
        embedding = reduct.fit_transform(X.T).T
        return embedding
    
    
    def execute(self, threshold):
        cov_model = self.get_covariance(self.X)
        self.cov = cov_model.covariance_
         
        self.cluster, self.labels = self.clustering(self.cov)
        
        partial_correlations = cov_model.precision_.copy()
        d = 1 / np.sqrt(np.diag(partial_correlations))
        partial_correlations *= d
        partial_correlations *= d[:, np.newaxis]
        self.non_zero = (np.abs(np.triu(partial_correlations, k=1)) > threshold)
        self.values = np.abs(partial_correlations[self.non_zero])
        
        
    def view(self, threshold = 0.1):
        self.liste = list(self.X.columns)
        self.embedding = self.dimension_reduction(self.X, n_comp = 2)
        self.df = pd.DataFrame(self.embedding.T , columns=['x', 'y'])
        self.df.index = self.liste
        self.df['labels'] = self.labels
        
        
        n_labels = self.labels.max()+1
        val_max = self.values.max()
        
        color_list = pl.cm.jet(np.linspace(0, 1, n_labels))
        my_color = [color_list[i] for i in self.labels]
        
        fig = plt.figure(figsize=(20, 10))
        ax = plt.axes([0, 0, 1, 1])
        plt.scatter(self.embedding[0], self.embedding[1], c=my_color)

        start_idx , end_idx = np.where(self.non_zero)
        segments = [[self.embedding[:,start] , self.embedding[:,end]] for start, end in zip(start_idx, end_idx)]
        lc = LineCollection(segments)
        lc.set_array(self.values)
        temp = 20*self.values
        temp2 = np.repeat(5, len(temp))
        w = np.minimum(temp, temp2)
        lc.set_linewidth(temp)
        ax.add_collection(lc)
        fig.colorbar(lc)

        for index, (self.liste, label, (x, y)) in enumerate(zip(self.liste, self.labels, self.embedding.T)):
            plt.text(x, y, self.liste, size = 10,
                    bbox = dict(facecolor='w', alpha=.8,
                                edgecolor = pl.cm.nipy_spectral(label/float(n_labels)))
                    )

        plt.xlim(self.embedding[0].min() - .15 * self.embedding[0].ptp(),
                self.embedding[0].max() + .15 * self.embedding[0].ptp())
        plt.ylim(self.embedding[1].min() - .03 * self.embedding[1].ptp(),
                    self.embedding[1].max() + .03 * self.embedding[1].ptp())
    
    
    def view3d(self, threshold = 0.1):
        self.liste = list(self.X.columns)
        self.embedding = self.dimension_reduction(self.X, n_comp = 3)
        self.df = pd.DataFrame(self.embedding.T , columns=['x', 'y', 'z'])
        self.df.index = self.liste
        self.df['labels'] = self.labels
        
        fig = px.scatter_3d(self.df['x'], self.df['y'], self.df['z'], color = self.df['labels'])
        fig.update_layout(height = 700 , width =1000,
                        margin=dict(l=5, r=0, b=10, t=30, pad=1 ),
                        showlegend=True,
                        )
        fig.show()
        
        
    def plot(self):
        ret = self.data
        cum_ret = (self.data + 1).cumprod()
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True,
                            row_heights = [0.7, 0.3], vertical_spacing = 0.01)
        
        for color, symbol in enumerate(self.data.columns):
            fig.add_trace(
                go.Scatter(
                    x = cum_ret.index , y = cum_ret[symbol],
                    name = symbol,
                    #marker_color = color,
                ),
                row = 1 , col =1
            )

            fig.add_trace(
                go.Scatter(
                    x = ret.index , y = ret[symbol],
                    name = 'ret_'+symbol,
                    #marker_color = color,
                ),
                row = 2 , col = 1
            )
        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)
        fig.update_layout(height = 800 , width =1500)
            
        fig.show()

