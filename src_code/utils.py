import torch
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt

# determine device to run network on (runs on gpu if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, data_loader, test_loader, optimizer, criterion, n_epochs, classes=None, verbose=False): 
    losses = []
    for epoch in range(n_epochs): 
        net.train()
        for i, batch in enumerate(data_loader): 

            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(imgs)


            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())

            if verbose: 
                print("[%d/%d][%d/%d] loss = %f" % (epoch, n_epochs, i, len(data_loader), loss.item()))

        # evaluate performance on testset at the end of each epoch 
        print("[%d/%d]" %(epoch, n_epochs))
        eval_target_net(net, test_loader, classes=classes)

        #plt.plot(losses)
        #plt.show()

def train_attacker(attack_net, shadow_net, shadow_train, shadow_out, optimizer, criterion, n_epochs, k): 
    losses = []
    
    shadow_net.train()
    attack_net.eval()
    for epoch in range(n_epochs): 
        attack_net.train()
        total = 0
        correct = 0
        
        #train_top = np.array([])
        #train_top = []
        train_top = np.empty((0,2))
        out_top = np.empty((0,2))
        for i, ((train_imgs, _), (out_imgs, _)) in enumerate(zip(shadow_train, shadow_out)): 

            #######out_imgs = torch.randn(out_imgs.shape)
            mini_batch_size = train_imgs.shape[0]
            train_imgs, out_imgs = train_imgs.to(device), out_imgs.to(device)
            
            train_posteriors = F.softmax(shadow_net(train_imgs.detach()), dim=1)

            out_posteriors = F.softmax(shadow_net(out_imgs.detach()), dim=1)
            
            optimizer.zero_grad()
            
            train_sort, _ = torch.sort(train_posteriors, descending=True)
            train_top_k = train_sort[:,:k].clone().to(device)

            out_sort, _ = torch.sort(out_posteriors, descending=True)
            out_top_k = out_sort[:,:k].clone().to(device)
            
            train_top = np.vstack((train_top,train_top_k[:,:2].cpu().detach().numpy()))
            out_top = np.vstack((out_top, out_top_k[:,:2].cpu().detach().numpy()))
            
            #print("train_top_k = ",train_top_k)
            #print("out_top_k = ",out_top_k)
            
            
            train_lbl = torch.ones(mini_batch_size).to(device)
            out_lbl = torch.zeros(mini_batch_size).to(device)
            
            
            train_predictions = torch.squeeze(attack_net(train_top_k))
            loss_train = criterion(train_predictions, train_lbl)
            loss_train.backward()
            optimizer.step()
            
            
            
            optimizer.zero_grad()
            
            out_predictions = torch.squeeze(attack_net(out_top_k))
            loss_out = criterion(out_predictions, out_lbl)
            loss_out.backward()
            optimizer.step()
            
            #print("train_predictions = ",train_predictions)
            #print("out_predictions = ",out_predictions)
            
            
            loss = (loss_train + loss_out) / 2 
            '''
            loss_train = criterion(train_predictions, train_lbl)
            loss_out = criterion(out_predictions, out_lbl)
            loss = (loss_train + loss_out) / 2 
            loss.backward() 
            optimizer.step()
            '''
            
            
            correct += (train_predictions>=0.5).sum().item()
            correct += (out_predictions<0.5).sum().item()
            total += train_predictions.size(0) + out_predictions.size(0)
            
            
            print("[%d/%d][%d/%d] loss = %.2f, accuracy = %.2f" % (epoch, n_epochs, i, len(shadow_train), loss.item(), 100 * correct / total))
        

        '''
        plt.scatter(out_top.T[0,:], out_top.T[1,:], c='b')
        plt.scatter(train_top.T[0,:], train_top.T[1,:], c='r')
        plt.show()
        '''
        
def eval_target_net(net, testloader, classes=None): 
    
    if classes is not None: 
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
    total = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for i, (imgs, lbls) in enumerate(testloader):
            
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            output = net(imgs)
            
            predicted = output.argmax(dim=1)
            
            total += imgs.size(0)
            correct += predicted.eq(lbls).sum().item()
            
            if classes is not None: 
                for prediction, lbl in zip(predicted, lbls): 

                    class_correct[lbl] += prediction == lbl
                    class_total[lbl] += 1 
    
    if classes is not None: 
        for i in range(len(classes)):
            print('Accuracy of %s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print("\nTotal accuracy = %.2f %%\n\n" % (100*(correct/total)) )
    
def eval_attack_net(attack_net, target_net, target_train, target_out, k): 
    losses = []
    
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

        train_posteriors = F.softmax(target_net(train_imgs.detach()), dim=1)
        out_posteriors = F.softmax(target_net(out_imgs.detach()), dim=1)

        train_sort, _ = torch.sort(train_posteriors, descending=True)
        train_top_k = train_sort[:,:k].clone().to(device)

        out_sort, _ = torch.sort(out_posteriors, descending=True)
        out_top_k = out_sort[:,:k].clone().to(device)

        train_top = np.vstack((train_top,train_top_k[:,:2].cpu().detach().numpy()))
        out_top = np.vstack((out_top, out_top_k[:,:2].cpu().detach().numpy()))

        #print("train_top_k = ",train_top_k)
        #print("out_top_k = ",out_top_k)


        train_lbl = torch.ones(mini_batch_size).to(device)
        out_lbl = torch.zeros(mini_batch_size).to(device)

        
        train_predictions = torch.squeeze(attack_net(train_top_k))
        out_predictions = torch.squeeze(attack_net(out_top_k))

        #print("train_predictions = ",train_predictions)
        #print("out_predictions = ",out_predictions)
        
        
        true_positives += (train_predictions >= 0.5).sum().item()
        false_positives += (out_predictions >= 0.5).sum().item()
        false_negatives += (train_predictions < 0.5).sum().item()
        

        correct += (train_predictions>=0.5).sum().item()
        correct += (out_predictions<0.5).sum().item()
        total += train_predictions.size(0) + out_predictions.size(0)

    accuracy = 100 * correct / total
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives !=0 else 0
    print("accuracy = %.2f, precision = %.2f, recall = %.2f" % (accuracy, precision, recall))
    
    

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

        print("threshold = %.4f, accuracy = %.2f, precision = %.2f, recall = %.2f" % (t, accuracy, precision, recall))


    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    