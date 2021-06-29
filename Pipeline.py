from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
from numpy.linalg import *

le = preprocessing.LabelEncoder()

def clean_data(X):
    le = LabelEncoder()
    categorical = []
    error = []
    missing = X[X.isna().sum()[X.isna().sum()>0].index].dtypes[X[X.isna().sum()[X.isna().sum()>0].index].dtypes=="float64"].index
    for col in missing:
        X[col] = X[col].fillna(np.mean(X[col]))
    for i in X.columns:
        if X[i].dtypes=="O":
            categorical.append(i)
    for i in range(0, len(categorical)):
        if X[categorical[i]].isna().sum() == 0:
            X[categorical[i]] = le.fit_transform(X[categorical[i]])
        else:
            error.append(categorical[i])
    X[error] = X[error].fillna("No such thing")
    for i in range(0, len(error)):
        X[error][i] = le.fit_transform(X[error][i])
    return X

def graphique_time(X):
    
    print("We can see that there is a great correlation between the type of trader and the average time per operation")
    
    names = pd.get_dummies(X["type"]).columns
    values = [pd.get_dummies(X["type"]).iloc[:, i].sum() for i in range(len(pd.get_dummies(X["type"]).columns))]
    second_time = [X[X["type"]==i]["NbSecondWithAtLeatOneTrade"].mean() for i in list(pd.get_dummies(X["type"]).columns)]

    fig, ax1 = plt.subplots(figsize=(15, 10))
    plt.title("Average second/Type of Trader", fontsize=15)
    s1 = values
    ax1.bar(names, values)
    ax1.set_xlabel('Type of trader', fontsize=15)
    ax1.set_ylabel('Number of trader', color='b', fontsize=15);
    ax2 = ax1.twinx()
    s2 = second_time
    ax2.plot(second_time, 'o', color='red')
    ax2.annotate("480.7", xy=(200, 545), xycoords='figure pixels', color="red")
    ax2.annotate("364.8", xy=(468, 415), xycoords='figure pixels', color="red")
    ax2.annotate("53.5", xy=(745, 53), xycoords='figure pixels', color="red")
    ax2.set_ylabel('Average second', color='r', fontsize=15);
    
def graphique_normal(X):
    fig = plt.figure(figsize=(20, 15))
    sns.set(font_scale=1.5)

    # (Corr= 0.817185) Box plot overallqual/salePrice
    fig4 = fig.add_subplot(221);
    sns.boxplot(x='type', y='NbTradeVenueMic', data=X[['NbTradeVenueMic', 'type']])

    # (Corr= 0.700927) GrLivArea vs SalePrice plot
    fig2 = fig.add_subplot(222); 
    sns.scatterplot(x = X[X["OTR"]<=5000]["OTR"], y = X[X["NbSecondWithAtLeatOneTrade"]<=9000]["NbSecondWithAtLeatOneTrade"], hue=X.type, palette= 'Spectral')

    # (Corr= 0.700927) GrLivArea vs SalePrice plot
    fig3 = fig.add_subplot(223); 
    sns.scatterplot(x = X["NbTradeVenueMic"], y = X["NbSecondWithAtLeatOneTrade"], hue=X.type, palette= 'Spectral')
    
    fig1 = fig.add_subplot(224);
    sns.boxplot(x="type", y="mean_time_two_events", data=X[["mean_time_two_events", "type"]])

    plt.tight_layout(); plt.show()
    

class PCA_Analysis:
    
    def __init__(self, number_of_components):
        
        self.nb = number_of_components
        
    def get_defra_scaled(self):
        
        train = pd.read_csv("X_train.csv")
        Y_train = pd.read_csv("Y_train.csv")
        df_merged = pd.merge(left=train, right=Y_train, left_on=["Trader"], right_on=["Trader"])
        index = [2*i + 1 for i in range(df_merged.shape[0]//2)]
        df_merged.drop(index, inplace=True)
        df_merged = clean_data(df_merged)
        
        df_PCA = df_merged.drop(["Trader", "Index", "type"], axis=1)
        defra_scaled = pd.DataFrame(StandardScaler().fit_transform(df_PCA))

        defra_scaled.columns = df_merged.drop(["Trader", "Index", "type"], axis=1).columns
        
        return defra_scaled
    
    def PCA_graph_3D(self):
        
        n_components = self.nb

        pca = PCA(n_components = n_components)
        pca.fit(self.get_defra_scaled())
        Projection = pca.transform(self.get_defra_scaled())
        
        Projection_List = Projection[:, 0], Projection[:, 1], Projection[:, 2]


        print("The variance explained by the {} main axes is : {:.3f} %".format(n_components, 100*sum(pca.explained_variance_ratio_)))

        print("\n")

        defra_pca = pca.fit_transform(self.get_defra_scaled())
        Label = self.get_defra_scaled().columns

        fig = figure(figsize = (35, 35))
        ax = Axes3D(fig)

        for i in range(1, n_components):
            ax.scatter(Projection_List[0][i], Projection_List[1][i], Projection_List[2][i], color='b') 
            ax.text(Projection_List[0][i], Projection_List[1][i], Projection_List[2][i],  '%s' % (Label[i]), size=25, zorder=1,  
    color='k')
            ax.set_xlabel("PC1", fontsize = 40)
            ax.set_ylabel("PC2", fontsize = 40)
            ax.set_zlabel("PC3", fontsize = 40)

    
        plt.title("Projection of the n points on the span of the three main axes", fontsize = 40);
        


        D, V = eig(self.get_defra_scaled().T @ self.get_defra_scaled()/len(self.get_defra_scaled()))
        index = np.argsort(D)[::-1][:n_components]
        importance = self.get_defra_scaled().columns[index]
        
        print("The {} most important axes are :{}".format(n_components, list(importance)))
        
    def PCA_graph_2D(self):
        
        n_components = self.nb
        
        A = np.dot(self.get_defra_scaled().T, self.get_defra_scaled())

        D, V = eig(A)

        Projection_1 = np.dot(self.get_defra_scaled(), V[:,0])
        Projection_2 = np.dot(self.get_defra_scaled(), V[:,2])

        plt.figure(figsize = (20, 20))
        
        Label = self.get_defra_scaled().columns

        for i in range(n_components):
                x, y = Projection_1[i], Projection_2[i]
                plt.text(Projection_1[i], Projection_2[i], Label[i], fontsize = 15)
                plt.scatter(x, y)
                plt.title("Projection of the n points on the span of the two main axes", fontsize = 25)
                plt.xlabel("PC1", fontsize = 25)
                plt.ylabel("PC2", fontsize = 25)
        
        
    
    def PCA_Variance(self):
        
        n_components = self.nb
        
        D, V = eig(self.get_defra_scaled().T @ self.get_defra_scaled()/len(self.get_defra_scaled()))
        index = np.argsort(D)[::-1][:n_components]
        importance = self.get_defra_scaled().columns[index]
       
        values = 100 * np.cumsum(D[index]/sum(D))
        names = np.arange(1, len(values)+1, 1)
        
        plt.figure(figsize=(15, 12))
        plt.title("Variance explained for the number of axis chosen")
        plt.xlabel("Number of axis chosen")
        plt.ylabel("Variance explained (in %)")
        plt.bar(names, values) ; plt.show()
        
        
    
    
