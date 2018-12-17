from .train import train, train_attacker
from .metrics import eval_membership_inference, eval_attack_model


def ml_leaks1(target=None, shadow_model=None, attacker_model=None,
              target_train_loader=None, target_out_loader=None,
              shadow_train_loader=None, shadow_out_loader=None,
              shadow_optim=None, attack_optim=None, shadow_criterion=None,
              attack_criterion=None, n_epochs=0, classes=None,
              n_max_posteriors=3, verbose=False):
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
        target_train_loader (DataLoader): Loads data used for target network
            training (split[0]).  This will be used for evaluation of the
            attack.
        target_out_loader (DataLoader): Loads data pointing to target out-of-
            training dataset (split[1]) used for attack evaluation.
        shadow_train_loader (DataLoader): Loader for shadow_model training
            (split[2]).
        shadow_out_loader: Out-of-sample from shadow net, used to train the
            attacker (split[3]).
        test_loader: Loader pointing to the validation data set, i.e. out of
            all samples training (split[4]).
        shadow_optim (torch.optim): Optimizer for shadow_model training.
        attack_optim (torch.optim): Optimizer for attacker_model training.
        shadow_criterion (torch.nn): Loss function for shadow_model training.
        attack_criterion (torch.nn): Loss function for attacker_model
            training.
        n_epochs (int): Number of epockhs used for shadow and attack net
            training.
        classes (list): Classes for membership inference task.
        n_max_posteriors (int): Number of maximal posteriors to use in
            membership inference attack.
        verbose (bool): If True will print the loss at each batch during all
            training steps.

    Example:

    To-do:
         Add example to docstring.
    '''
    print('---- Training shadow network ----')
    train(model=shadow_model, data_loader=shadow_train_loader,
          test_loader=shadow_out_loader, optimizer=shadow_optim,
          criterion=shadow_criterion, n_epochs=n_epochs, classes=classes,
          verbose=verbose)
    #
    print('---- Training attack network ----')
    train_attacker(attack_model=attacker_model, shadow=shadow_model,
                   shadow_train=shadow_train_loader,
                   shadow_out=shadow_out_loader, optimizer=attack_optim,
                   criterion=attack_criterion, n_epochs=n_epochs,
                   k=n_max_posteriors)
    #
    print('---- Evaluate attack ----')
    eval_attack_model(attack_model=attacker_model, target=target,
                      target_train=target_train_loader,
                      target_out=target_out_loader, k=n_max_posteriors)


def ml_leaks3(target=None, target_train_loader=None,  target_out_loader=None):
    ''' Implementation of  ml_leaks 3 membership inference attack

    Args:
         target (nn.Module): Trained target network to attack
         target_train_loader (DataLoader): Loader pointing to data used to
              train target (split[0]).  Used here to evaluate attack
              performance.
         target_out_loader: Loader pointing to the target out-of-training data
              (split[1])

    Example:

    To-do:
         Add example to docstring.
    '''
    eval_membership_inference(target_model=target,
                              target_train=target_train_loader,
                              target_out=target_out_loader)
