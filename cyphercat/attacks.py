# Pytorch imports
import torch
# Cyphercat imports
from .train import train, train_attacker
from .metrics import eval_membership_inference, eval_attack_model
# Device to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ml_leaks1(target=None, shadow_model=None, attacker_model=None,
              target_in_loader=None, target_out_loader=None,
              shadow_train_loader=None, shadow_out_loader=None,
              shadow_optim=None, attack_optim=None, shadow_criterion=None,
              attack_criterion=None, shadow_epochs=0, attack_epochs=0,
              classes=None, n_max_posteriors=3, retrain=True, verbose=False):
    '''Implementation of ml_leaks 1 membership inference attack

    Trains shadow network an independent data set  and then trains the
    attacker to infer membership on this shadow net. Finally, the attacker is
    used to run mberbeship inference on the target.

    Args:
        target (nn.Module): Trained target network.
        shadow_model (nn.Module): Shadow network to help train the attacker in
            membership inference task.
        attacker_model (nn.Module): Network to be trained in membership
            inference task.
        target_in_loader (DataLoader): DataLoader pointing to target in-data
            used for testing the attack (split[4])
        target_out_loader (DataLoader): Loads data pointing to target out-of-
            training dataset (split[1]) used for attack evaluation.
        shadow_train_loader (DataLoader): Loader for shadow_model training
            (split[2]).
        shadow_out_loader: Out-of-sample from shadow net, used to train the
            attacker (split[3]).
        shadow_optim (torch.optim): Optimizer for shadow_model training.
        attack_optim (torch.optim): Optimizer for attacker_model training.
        shadow_criterion (torch.nn): Loss function for shadow_model training.
        attack_criterion (torch.nn): Loss function for attacker_model
            training.
        shadow_epochs (int): Number of epochs used to train the shadow network.
        attack_epochs (int): Number of epochs used to train the attack network.
        classes (list): Classes for membership inference task.
        n_max_posteriors (int): Number of maximal posteriors to use in
            membership inference attack.
        retrain (bool): If True will retrain the shadow and attack network,
            otherwise will simply use the provided attacker model as is fed.
        verbose (bool): If True will print the loss at each batch during all
            training steps.

    Example:

    To-do:
         Add example to docstring.
    '''
    if retrain:
        print('---- Training shadow network ----')
        train(model=shadow_model, data_loader=shadow_train_loader,
              test_loader=shadow_out_loader, optimizer=shadow_optim,
              criterion=shadow_criterion, n_epochs=shadow_epochs,
              classes=classes, verbose=verbose)
        #
        print('---- Training attack network ----')
        train_attacker(attack_model=attacker_model, shadow_model=shadow_model,
                       shadow_train=shadow_train_loader,
                       shadow_out=shadow_out_loader, optimizer=attack_optim,
                       criterion=attack_criterion, n_epochs=attack_epochs,
                       k=n_max_posteriors)
    #
    print('---- Evaluate attack ----')
    df_pr = eval_attack_model(attack_model=attacker_model, target=target,
                              target_train=target_in_loader,
                              target_out=target_out_loader, k=n_max_posteriors)

    return df_pr


def ml_leaks3(target=None, target_in_loader=None,  target_out_loader=None):
    ''' Implementation of  ml_leaks 3 membership inference attack

    Args:
        target (nn.Module): Trained target network to attack
        target_in_loader (DataLoader): Loader pointing to data used to
            train target (split[4]).  Used here to evaluate attack
            performance.
        target_out_loader: Loader pointing to the target out-of-training data
            (split[1])

    Example:

    To-do:
        Add example to docstring.
    '''
    eval_membership_inference(target_model=target,
                              target_train=target_in_loader,
                              target_out=target_out_loader)


def mi_gradient_ascent(input_sample=None, target_model=None, optimizer=None,
                       category=None, iterations=0, verbose=False):
    """ Implementation of gradient based model inversion attack

    Args:
        input_sample (torch.tensor): Initialized input sample, usually
            randomly generated. Size should match the model input.
        target_model (nn.Module): Pretrained model to attack.
        optimizer (nn.optim): Optimizer (initialized on image parameters) used
            in attack.
        category (int): Category to invert.
        iterations (int): Query iterations in the attack.
        verbose (bool): If True will print the loss at each step in attack.

    Returns:
        (list(float)): Returns a list of the losses at each iteration.
    Example:

    Todos:
        Write example
    """
    category = torch.Variable(torch.LongTensor([category])).to(device)
    losses = []

    for i_step in range(iterations):
        target_model.zero_grad()
        out = target_model(input_sample)
        loss = -out.take(category)
        loss.backward()
        #
        optimizer.step()
        input_sample.grad.zero_()
        losses.append(loss.data)
        #

    return losses
