import numpy as np
import random
from sklearn import datasets
import matplotlib.pyplot as plt
iris=datasets.load_iris()
predictors=iris.data[:,0:2]
outcomes=iris.target#Zones:0 for Red,1 for Orange,2 for Green.
def distance(p1,p2):
    """Finds the distance between 2 points"""
    return np.sqrt(np.sum(np.power(p2-p1,2)))
def majority_vote(votes):
    """Finds the most common zone."""
    vote_count={}
    for vote in votes:
        if vote in vote_count:
            vote_count[vote]+=1
        else:
            vote_count[vote]=1
    winners=[]
    max_counts=max(vote_count.values())
    for vote, count in vote_count.items():
        if count==max_counts:
            winners.append(vote)
    return random.choice(winners)
def find_nearest_neighbours(p,points,k):
    """find the k nearest neighbours of point p and return ther index"""
    distances=np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i]=distance(p,points[i])
    ind=np.argsort(distances)#returns array of index positions of elements according to which it is sorted.
    return ind[:k]
def knn_predict(p,points,k,outcomes,ind):
    return majority_vote(outcomes[ind])
    # predict the class of p based on majority.
def make_prediction_grid(predictors,outcomes,limits,h,k,n):
    """classify each point on prediction grid"""
    (x_min,x_max,y_min,y_max)=limits
    xs=np.arange(x_min,x_max,h)
    ys=np.arange(y_min,y_max,h)
    xx,yy=np.meshgrid(xs,ys)
    prediction_grid=np.zeros(xx.shape)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p=np.array([x,y])
            prediction_grid[j,i]=n
    return(xx,yy,prediction_grid)
def plot_prediction_grid (xx, yy, prediction_grid):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","orange", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green","blue"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
def execute():
    p=np.array([5.5,3])
    k=10
    limits=(4,8,1.5,4.5)
    h=0.1
    zone=""
    ind=find_nearest_neighbours(p,predictors,k)
    n=majority_vote(outcomes[ind])
    (xx,yy,prediction_grid)=make_prediction_grid(predictors,outcomes,limits,h,k,n)
    plot_prediction_grid(xx,yy,prediction_grid)
    plt.figure()
    plt.plot(predictors[outcomes==0][:,0],predictors[outcomes==0][:,1],"ro")
    plt.plot(predictors[outcomes==2][:,0],predictors[outcomes==2][:,1],"go")
    plt.plot(predictors[outcomes==1][:,0],predictors[outcomes==1][:,1],"o",color='orange')
    plt.plot(p[0],p[1],"bo")
    '''plt.savefig('static/images/imag1.png')'''
    '''plt.show()'''
    if(n==0):
        zone="RED. Please do not leave ypur house."
    elif(n==1):
        zone="ORANGE. Please leave your house only in case of an emergency."
    else:
        zone="GREEN. You are free to go out but be careful."
    '''print("The Entered Coordinate is in color Blue")'''
    '''print("The given Coordinate is in the given zone:",zone)'''
    return zone

