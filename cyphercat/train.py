
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fcnal

from .metrics import eval_target_model

# Determine device to run network on (runs on gpu if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model=None, data_loader=None, test_loader=None,
          optimizer=None, criterion=None, n_epochs=0,
          classes=None, verbose=False):
    """
    Function to train a model provided
    specified train/test sets and associated
    training parameters.

    Parameters
    ----------
    model       : Module
                  PyTorch conforming nn.Module function
    data_loader : DataLoader
                  PyTorch dataloader function
    test_loader : DataLoader
                  PyTorch dataloader function
    optimizer   : opt object
                  PyTorch conforming optimizer function
    criterion   : loss object
                  PyTorch conforming loss function
    n_epochs    : int
                  number of training epochs
    classes     : list
                  list of classes
    verbose     : boolean
                  flag for verbose print statements
    """
    losses = []
    for epoch in range(n_epochs):
        model.train()
        for i, batch in enumerate(data_loader):

            data, labels = batch
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if verbose:
                print("[{}/{}][{}/{}] loss = {}"
                      .format(epoch, n_epochs, i,
                              len(data_loader), loss.item()))

        # evaluate performance on testset at the end of each epoch
        print("[{}/{}]".format(epoch, n_epochs))
        print("Training:")
        train_acc = eval_target_model(model, data_loader, classes=classes)
        print("Test:")
        test_acc = eval_target_model(model, test_loader, classes=classes)
        # plt.plot(losses)
        # plt.show()


def train_attacker(attack_model=None, shadow_model=None,
                   shadow_train=None, shadow_out=None,
                   optimizer=None, criterion=None, n_epochs=0, k=0):
    """
    Trains attack model (classifies a sample as in or
    out of training set) using shadow model outputs
    (probabilities for sample class predictions).
    The type of shadow model used can vary.

    Parameters
    ----------
    attack_model : Module
                   PyTorch conforming nn.Module function
    shadow_model : Module
                   PyTorch conforming nn.Module function
    shadow_train : DataLoader
                   PyTorch dataloader function
    shadow_out   : DataLoader
                   PyTorch dataloader function
    optimizer    : opt object
                   PyTorch conforming optimizer function
    criterion    : loss object
                   PyTorch conforming loss function
    n_epochs     : int
                   number of training epochs
    k            : int
                   Value at which to end using train data list
    """

    in_predicts = []
    out_predicts = []

    if type(shadow_model) is not Pipeline:
        shadow_model = shadow_model
        shadow_model.eval()

    for epoch in range(n_epochs):

        total = 0
        correct = 0

        train_top = np.empty((0, 2))
        out_top = np.empty((0, 2))
        for i, ((train_data, _), (out_data, _)) in enumerate(zip(shadow_train, shadow_out)):

            # out_data = torch.randn(out_data.shape)
            mini_batch_size = train_data.shape[0]

            if type(shadow_model) is not Pipeline:
                train_data, out_data = train_data.to(device), out_data.to(device)
                train_posteriors = fcnal.softmax(shadow_model(train_data.detach()), dim=1)
                out_posteriors = fcnal.softmax(shadow_model(out_data.detach()), dim=1)

            else:
                traininputs = train_data.view(train_data.shape[0], -1)
                outinputs = out_data.view(out_data.shape[0], -1)

                in_preds = shadow_model.predict_proba(traininputs)
                train_posteriors = torch.from_numpy(in_preds).float()
                # for p in in_preds:
                #     in_predicts.append(p.max())

                out_preds = shadow_model.predict_proba(outinputs)
                out_posteriors = torch.from_numpy(out_preds).float()
                # for p in out_preds:
                #    out_predicts.append(p.max())

            train_sort, _ = torch.sort(train_posteriors, descending=True)
            train_top_k = train_sort[:, :k].clone().to(device)
            for p in train_top_k:
                in_predicts.append((p.max()).item())
            out_sort, _ = torch.sort(out_posteriors, descending=True)
            out_top_k = out_sort[:, :k].clone().to(device)
            for p in out_top_k:
                out_predicts.append((p.max()).item())

            train_top = np.vstack((train_top, train_top_k[:, :2].cpu().detach().numpy()))
            out_top = np.vstack((out_top, out_top_k[:, :2].cpu().detach().numpy()))

            train_lbl = torch.ones(mini_batch_size).to(device)
            out_lbl = torch.zeros(mini_batch_size).to(device)

            optimizer.zero_grad()

            train_predictions = torch.squeeze(attack_model(train_top_k))
            out_predictions = torch.squeeze(attack_model(out_top_k))

            loss_train = criterion(train_predictions, train_lbl)
            loss_out = criterion(out_predictions, out_lbl)

            loss = (loss_train + loss_out) / 2

            if type(shadow_model) is not Pipeline:
                loss.backward()
                optimizer.step()

            correct += (fcnal.sigmoid(train_predictions) >= 0.5).sum().item()
            correct += (fcnal.sigmoid(out_predictions) < 0.5).sum().item()
            total += train_predictions.size(0) + out_predictions.size(0)

            print("[{}/{}][{}/{}] loss = {:.2f}, accuracy = {:.2f}"
                  .format(epoch, n_epochs, i, len(shadow_train),
                          loss.item(), 100 * correct / total))

        # Plot distributions for target predictions
        # in training set and out of training set
        """
        fig, ax = plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.hist(in_predicts, bins='auto')
        plt.title('In')
        plt.subplot(2,1,2)
        plt.hist(out_predicts, bins='auto')
        plt.title('Out')
        """

        '''
        plt.scatter(out_top.T[0,:], out_top.T[1,:], c='b')
        plt.scatter(train_top.T[0,:], train_top.T[1,:], c='r')
        plt.show()
        '''


class softCrossEntropy(torch.nn.Module):
    def __init__(self, alpha=0.95):
        """
        :param alpha: Strength (0-1) of influence from soft labels in training
        """
        super(softCrossEntropy, self).__init__()
        self.alpha = alpha
        return

    def forward(self, inputs, target, true_labels):
        """
        :param inputs: predictions
        :param target: target (soft) labels
        :param true_labels: true (hard) labels
        :return: loss
        """

        KD_loss = self.alpha
        KD_loss *= nn.KLDivLoss(size_average=False)(
                                 fcnal.log_softmax(inputs, dim=1),
                                 fcnal.softmax(target, dim=1)
                                )
        KD_loss += (1-self.alpha)*fcnal.cross_entropy(inputs, true_labels)

        return KD_loss


def distill_training(teacher=None, learner=None, data_loader=None,
                     test_loader=None, optimizer=None,
                     criterion=None, n_epochs=0, verbose=False):
    """
    :param teacher: network to provide soft labels in training
    :param learner: network to distill knowledge into
    :param data_loader: data loader for training data set
    :param test_loaderL data loader for validation data
    :param optimizer: optimizer for training
    :param criterion: objective function, should allow for soft labels.
                      We suggest softCrossEntropy
    :param n_epochs: epochs for training
    :param verbose: verbose == True will print loss at each batch
    :return: None, teacher model is trained in place
    """
    losses = []
    for epoch in range(n_epochs):
        teacher.eval()
        learner.train()
        for i, batch in enumerate(data_loader):
            with torch.set_grad_enabled(False):
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                soft_lables = teacher(data)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                outputs = learner(data)
                loss = criterion(outputs, soft_lables, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if verbose:
                    print("[{}/{}][{}/{}] loss = {}"
                          .format(epoch, n_epochs, i,
                                  len(data_loader), loss.item()))

        # evaluate performance on testset at the end of each epoch
        print("[{}/{}]".format(epoch, n_epochs))

        print("Training:")
        train_acc = eval_target_model(learner, data_loader, classes=None)

        print("Testing:")
        test_acc = eval_target_model(learner, test_loader, classes=None)
