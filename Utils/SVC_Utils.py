import numpy as np 
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

def load(dataloader):
    for data in dataloader:
        x,y=data
    x=x.view(x.shape[0],-1)
    return x,y

def hp_grid(n_components, C_range, gamma_range):
    clfs=[]
    pca=PCA(n_components=n_components)
    scaling = MinMaxScaler(feature_range=(-1,1))
    for i in C_range:
        for j in gamma_range:
            svc=svm.SVC(C=i, gamma=j)
            clf=make_pipeline(pca, scaling, svc)
            clfs.append(clf)
    return clfs

def train_grid(clfs, inputs, targets):
    fitted_clfs=[]
    for i in range(len(clfs)):
        x=clfs[i].fit(inputs, targets)
        fitted_clfs.append(x)
        print('Fitted: ', i+1, '/', len(clfs))
    return fitted_clfs

def predict_eval(clfs, inputs, targets, training=False):
    for i in clfs:
        preds=i.predict(inputs)
        num_correct=torch.eq(torch.from_numpy(preds), targets).sum().item()
        acc=(num_correct/len(targets))*100
        if training:
            print('C: ', i.get_params(deep=True)['C'], 'gamma: ', i.get_params(deep=True)['gamma'])
            print('Training Accuracy: ', acc)
        else:
            print('Testing Accuracy: ', acc)
        return preds, acc

def save(fn_maxacc, fn_gen, test_accs, train_accs, fitted_clfs):
    test=np.array(test_accs); train=np.array(train_accs)
    joblib.dump(fitted_clfs[np.argmax(test)], fn_maxacc)
    joblib.dump(fitted_clfs[np.argmin(train-test)], fn_gen)
    
