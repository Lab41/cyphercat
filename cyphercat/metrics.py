
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from .utils.svc_utils import *

# determine device to run network on (runs on gpu if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_target_net(model, data_loader, classes=None):

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
            print('Accuracy of {} : {:.2f} %%'.format(classes[i], 100 * class_correct[i] / class_total[i]))
    print("\nAccuracy = {:.2f} %%\n\n".format(accuracy))
    
    return accuracy

def eval_attack_net(attack_net, target, target_train, target_out, k):
    """Assess accuracy, precision, and recall of attack model for in training set/out of training set classification.
    Edited for use with SVCs."""
    
    in_predicts=[]
    out_predicts=[]
    losses = []
    
    if type(target) is not Pipeline:
        target_net=target
        target_net.eval()
        
    attack_net.eval()

    total = 0
    correct = 0

    train_top = np.empty((0,2))
    out_top = np.empty((0,2))

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(target_train, target_out)):


        mini_batch_size = train_imgs.shape[0]
        train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)
        
        #[mini_batch_size x num_classes] tensors, (0,1) probabilities for each class for each sample)
        if type(target) is Pipeline:
            traininputs=train_imgs.view(train_imgs.shape[0], -1)
            outinputs=out_imgs.view(out_imgs.shape[0], -1)
            
            train_posteriors=torch.from_numpy(target.predict_proba(traininputs)).float()
            out_posteriors=torch.from_numpy(target.predict_proba(outinputs)).float()
            
        else:
            train_posteriors = F.softmax(target_net(train_imgs.detach()), dim=1)
            out_posteriors = F.softmax(target_net(out_imgs.detach()), dim=1)
        

        #[k x mini_batch_size] tensors, (0,1) probabilities for top k probable classes
        train_sort, _ = torch.sort(train_posteriors, descending=True)
        train_top_k = train_sort[:,:k].clone().to(device)

        out_sort, _ = torch.sort(out_posteriors, descending=True)
        out_top_k = out_sort[:,:k].clone().to(device)
        
        #Collects probabilities for predicted class.
        for p in train_top_k:
            in_predicts.append((p.max()).item())
        for p in out_top_k:
            out_predicts.append((p.max()).item())
        
        if type(target) is not Pipeline:
            train_top = np.vstack((train_top,train_top_k[:,:2].cpu().detach().numpy()))
            out_top = np.vstack((out_top, out_top_k[:,:2].cpu().detach().numpy()))

        #print("train_top_k = ",train_top_k)
        #print("out_top_k = ",out_top_k)

        train_lbl = torch.ones(mini_batch_size).to(device)
        out_lbl = torch.zeros(mini_batch_size).to(device)
        #print(train_top_k)

        #Takes in probabilities for top k most likely classes, outputs ~1 (in training set) or ~0 (out of training set)
        train_predictions = F.sigmoid(torch.squeeze(attack_net(train_top_k)))
        out_predictions = F.sigmoid(torch.squeeze(attack_net(out_top_k)))

        #print("train_predictions = ",train_predictions)
        #print("out_predictions = ",out_predictions)


        true_positives += (train_predictions >= 0.5).sum().item()
        false_positives += (out_predictions >= 0.5).sum().item()
        false_negatives += (train_predictions < 0.5).sum().item()

        
        correct += (train_predictions>=0.5).sum().item()
        correct += (out_predictions<0.5).sum().item()
        total += train_predictions.size(0) + out_predictions.size(0)
    
    #Plot distributions for target predictions in training set and out of training set
    fig, ax = plt.subplots(2,1)
    plt.subplot(2,1,1)
    plt.hist(in_predicts, bins='auto')
    plt.title('In')
    plt.subplot(2,1,2)
    plt.hist(out_predicts, bins='auto')
    plt.title('Out')
    
    accuracy = 100 * correct / total
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives !=0 else 0

    print("accuracy = {:.2f}, precision = {:.2f}, recall = {:.2f}".format(accuracy, precision, recall))
    


def eval_membership_inference(target_net, target_train, target_out):

    target_net.eval()

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

    for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(target_train, target_out)):


        mini_batch_size = train_imgs.shape[0]
        train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)

        train_posteriors = F.softmax(target_net(train_imgs.detach()), dim=1)
        out_posteriors = F.softmax(target_net(out_imgs.detach()), dim=1)

        train_sort, _ = torch.sort(train_posteriors, descending=True)
        train_top = train_sort[:,0].clone().to(device)

        out_sort, _ = torch.sort(out_posteriors, descending=True)
        out_top = out_sort[:,0].clone().to(device)

        #print(train_top.shape)

        for j, t in enumerate(thresholds):
            true_positives[j] += (train_top >= t).sum().item()
            false_positives[j] += (out_top >= t).sum().item()
            false_negatives[j] += (train_top < t).sum().item()
            #print(train_top >= threshold)


            #print((train_top >= threshold).sum().item(),',',(out_top >= threshold).sum().item())

            correct[j] += (train_top >= t).sum().item()
            correct[j] += (out_top < t).sum().item()
            total[j] += train_top.size(0) + out_top.size(0)

    #print(true_positives,',',false_positives,',',false_negatives)

    for j, t in enumerate(thresholds):
        accuracy = 100 * correct[j] / total[j]
        precision = true_positives[j] / (true_positives[j] + false_positives[j]) if true_positives[j] + false_positives[j] != 0 else 0
        recall = true_positives[j] / (true_positives[j] + false_negatives[j]) if true_positives[j] + false_negatives[j] !=0 else 0
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        print("threshold = {:.4f}, accuracy = {:.2f}, precision = {:.2f}, recall = {:.2f}".format(t, accuracy, precision, recall))


    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
