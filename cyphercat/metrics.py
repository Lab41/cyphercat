
import torch
import torch.nn.functional as fcnal

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from .utils.svc_utils import *

# determine device to run network on (runs on gpu if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_target_model(model=None, data_loader=None, classes=None):
    """
    Function to evaluate a target model provided
    specified data sets.

    Parameters
    ----------
    model       : Module
                  PyTorch conforming nn.Module function
    data_loader : DataLoader
                  PyTorch dataloader function
    classes     : list
                  list of classes

    Returns
    -------
    accuracy    : float
                  accuracy of target model
    """

    if classes is not None:
        n_classes = len(classes)
        class_correct = np.zeros(n_classes)
        class_total   = np.zeros(n_classes)
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for i, (imgs, lbls) in enumerate(data_loader):

            imgs, lbls = imgs.to(device), lbls.to(device)

            output = model(imgs)

            predicted = output.argmax(dim=1)

            total += imgs.size(0)
            correct += predicted.eq(lbls).sum().item()

            if classes is not None:
                for prediction, lbl in zip(predicted, lbls):

                    class_correct[lbl] += prediction == lbl
                    class_total[lbl] += 1
                    
    accuracy = 100*(correct/total)
    if classes is not None:
        for i in range(len(classes)):
            print('Accuracy of {} : {:.2f} %%'
                  .format(classes[i], 
                          100 * class_correct[i] / class_total[i]))

    print("\nAccuracy = {:.2f} %%\n\n".format(accuracy))

    return accuracy


def eval_attack_model(attack_model=None, target=None, target_train=None, target_out=None, k=0):
    """
    Assess accuracy, precision, and recall of attack model 
    for in training set/out of training set classification.
    Edited for use with SVCs.

    Parameters
    ----------
    attack_model : Module
                   PyTorch conforming nn.Module function
    target       : Module
                   PyTorch conforming nn.Module function
    target_train : DataLoader
                   PyTorch dataloader function
    target_out   : DataLoader
                   PyTorch dataloader function
    k            : int
                   Value at which to end using train data list
    """

    
    in_predicts = []
    out_predicts = []
    losses = []
    
    if type(target) is not Pipeline:
        target_model = target
        target_model.eval()
        
    attack_model.eval()
    
    precisions = []
    recalls = []
    accuracies = []

    #for threshold in np.arange(0.5, 1, 0.005):
    thresholds = np.arange(0.5, 1, 0.005)

    total = np.zeros(len(thresholds))
    correct = np.zeros(len(thresholds))

    true_positives = np.zeros(len(thresholds))
    false_positives = np.zeros(len(thresholds))
    false_negatives = np.zeros(len(thresholds)) 
    
    train_top = np.empty((0,2))
    out_top = np.empty((0,2))

    for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(target_train, target_out)):


        mini_batch_size = train_imgs.shape[0]
        train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)
        
        # [mini_batch_size x num_classes] tensors, (0,1) probabilities for each class for each sample)
        if type(target) is Pipeline:
            traininputs = train_imgs.view(train_imgs.shape[0], -1)
            outinputs = out_imgs.view(out_imgs.shape[0], -1)
            
            train_posteriors = torch.from_numpy(target.predict_proba(traininputs)).float()
            out_posteriors = torch.from_numpy(target.predict_proba(outinputs)).float()
            
        else:
            train_posteriors = fcnal.softmax(target_model(train_imgs.detach()), dim=1)
            out_posteriors = fcnal.softmax(target_model(out_imgs.detach()), dim=1)
        

        # [k x mini_batch_size] tensors, (0,1) probabilities for top k probable classes
        train_sort, _ = torch.sort(train_posteriors, descending=True)
        train_top_k = train_sort[:, :k].clone().to(device)

        out_sort, _ = torch.sort(out_posteriors, descending=True)
        out_top_k = out_sort[:, :k].clone().to(device)
        
        # Collects probabilities for predicted class.
        for p in train_top_k:
            in_predicts.append((p.max()).item())
        for p in out_top_k:
            out_predicts.append((p.max()).item())
        
        if type(target) is not Pipeline:
            train_top = np.vstack((train_top, train_top_k[:, :2].cpu().detach().numpy()))
            out_top = np.vstack((out_top, out_top_k[:, :2].cpu().detach().numpy()))

        train_lbl = torch.ones(mini_batch_size).to(device)
        out_lbl = torch.zeros(mini_batch_size).to(device)

        # Takes in probabilities for top k most likely classes, 
        # outputs ~1 (in training set) or ~0 (out of training set)
        train_predictions = fcnal.sigmoid(torch.squeeze(attack_model(train_top_k)))
        out_predictions = fcnal.sigmoid(torch.squeeze(attack_model(out_top_k)))

        for j, t in enumerate(thresholds):
            true_positives[j] += (train_predictions >= t).sum().item()
            false_positives[j] += (out_predictions >= t).sum().item()
            false_negatives[j] += (train_predictions < t).sum().item()

            correct[j] += (train_predictions >= t).sum().item()
            correct[j] += (out_predictions < t).sum().item()
            total[j] += train_predictions.size(0) + out_predictions.size(0)
    
    for j, t in enumerate(thresholds):
        accuracy = 100 * correct[j] / total[j]
        precision = true_positives[j] / (true_positives[j] + false_positives[j]) if true_positives[j] + false_positives[j] != 0 else 0
        recall = true_positives[j] / (true_positives[j] + false_negatives[j]) if true_positives[j] + false_negatives[j] !=0 else 0
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        print("threshold = %.4f, accuracy = %.2f, precision = %.2f, recall = %.2f" % (t, accuracy, precision, recall))
    
#     accuracy = 100 * correct / total
#     precision = 0
#     if true_positives + false_positives != 0:
#         precision = true_positives / (true_positives + false_positives)

#     recall = 0
#     if true_positives + false_negatives != 0:
#         recall = true_positives / (true_positives + false_negatives)

#     print("accuracy = {:.2f} %%\nprecision = {:.2f} \nrecall = {:.2f}".format(accuracy, precision, recall))
    
    #Make a dataframe of precision & recall results
    df_pr = pd.DataFrame(columns =['Thresholds','Accuracy','Precision','Recall'], data = np.transpose([thresholds,accuracies,precisions,recalls]))
    return df_pr



def eval_membership_inference(target_model=None, target_train=None, target_out=None):
    """
    Function to evaluate a target model for 
    membership inference.

    Parameters
    ----------
    target_model : Module
                   PyTorch conforming nn.Module function
    target_train : DataLoader
                   PyTorch dataloader function
    target_out   : DataLoader
                   PyTorch dataloader function
    """

    target_model.eval()

    precisions = []
    recalls = []
    accuracies = []

    thresholds = np.arange(0.5, 1, 0.005)

    total = np.zeros(len(thresholds))
    correct = np.zeros(len(thresholds))

    true_positives = np.zeros(len(thresholds))
    false_positives = np.zeros(len(thresholds))
    false_negatives = np.zeros(len(thresholds))

    for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(target_train, target_out)):


        mini_batch_size = train_imgs.shape[0]
        train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)

        train_posteriors = fcnal.softmax(target_model(train_imgs.detach()), dim=1)
        out_posteriors = fcnal.softmax(target_model(out_imgs.detach()), dim=1)

        train_sort, _ = torch.sort(train_posteriors, descending=True)
        train_top = train_sort[:, 0].clone().to(device)

        out_sort, _ = torch.sort(out_posteriors, descending=True)
        out_top = out_sort[:, 0].clone().to(device)

        for j, t in enumerate(thresholds):
            true_positives[j] += (train_top >= t).sum().item()
            false_positives[j] += (out_top >= t).sum().item()
            false_negatives[j] += (train_top < t).sum().item()

            correct[j] += (train_top >= t).sum().item()
            correct[j] += (out_top < t).sum().item()
            total[j] += train_top.size(0) + out_top.size(0)

    for j, t in enumerate(thresholds):

        accuracy = 100 * correct[j] / total[j]

        precision = 0
        if true_positives[j] + false_positives[j] != 0:
            precision = true_positives[j] / (true_positives[j] + false_positives[j])

        recall = 0
        if true_positives[j] + false_negatives[j] != 0:
            recall = true_positives[j] / (true_positives[j] + false_negatives[j])

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        print("threshold = {:.4f}, accuracy = {:.2f}, precision = {:.2f}, recall = {:.2f}"
              .format(t, accuracy, precision, recall))

    # plt.plot(recalls, precisions)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.show()
    
    