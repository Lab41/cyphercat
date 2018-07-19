import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from metrics import * 

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

