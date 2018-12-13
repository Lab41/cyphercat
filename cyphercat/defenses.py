import copy
# Torch imports
import torch
import torch.optim as optim
# Local imports
from .train import train, softCrossEntropy
from .metrics import eval_target_model


def transfer_learn(model=None, data_loader=None, test_loader=None,
                   optimizer=None, criterion=None, lr=0, n_epochs=0,
                   unfreeze_layers=None, fine_tune=True, verbose=False):
    """Routine to perform transfer learning given a model and a new dataset.
    
    Args:
        model (nn.Module): Pretrained model.
        data_loader (Dataloader): Dataloader pointing to training dataset.
        test_loader (Dataloader): Dataloader poinitng to validation dataset.
        optimizer (config): This remains to be implemented
        criterion (nn.Module): Criterion for loss calculation.
        lr (float): Learning rate for training.
        n_epochs (int): Maximum number of epochs during training of last
            layers and fine-tunning step.
        unfreze_layers ((int, int)): Tuple with indices (first, last) of
            layers to unfreeze during first stage of training.
        fine_tune (bool): If true will do a second stage of training with all
            of the layers unfrozen and 1/10th of original learning rate.
        verbose (bool): If True will print loss at each training step.

    Returns:

    Todos:
        Implement generalized optimizer, loaded from configuration. Currentl
        hardcoded to SGD.
    """
    unfrozen = []
    param_list = list()
    for idx, mod in enumerate(model._modules.items()):
        if unfreeze_layers[0] <= idx <= unfreeze_layers[1]:
            param_list += list(mod[1].parameters())
            unfrozen.append(mod[0])
            for param in mod[1].parameters():
                param.requires_grad = True
        else:
            for param in mod[1].parameters():
                param.requires_grad = False

    print('Training parameters in modules:')
    for x in unfrozen:
        print('\t %s' % x)

    optimizer = optim.SGD(params=param_list, lr=lr, momentum=0.9)
    train(model=model, data_loader=data_loader, test_loader=test_loader,
          optimizer=optimizer, criterion=criterion, n_epochs=n_epochs,
          verbose=verbose)

    print('Finished training last layers, performance: \n'
          'Training: %lf \nTest: %lf' % (
              eval_target_model(model=model, data_loader=data_loader),
              eval_target_model(model=model, data_loader=test_loader)))

    if fine_tune is False:
        return

    print('Moving on to fine tunning entire network')
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.SGD(params=model.parameters(), lr=lr/10., momentum=0.9)
    train(model=model, data_loader=data_loader, test_loader=test_loader,
          optimizer=optimizer, criterion=criterion, n_epochs=n_epochs,
          verbose=verbose)

    print('Finished training last layers, performance: \n'
      'Training: %lf \nTest: %lf' % (
          eval_target_model(model=model, data_loader=data_loader),
          eval_target_model(model=model, data_loader=test_loader)))
    return

        
class dimensionality_reduction():
    """Returns a wrapped model that will return only the top-n_dim most
       probable classes during inference.
    """
    def __init__(self, model=None, n_top=1, break_posterior=False):
        """ Initializes the wrapped model.

        Args:
            model (nn.Module): Original, trained model to defend.
            n_dim (int): New dimensionality, i.e. the number of top ranked
                labels to return.
            break_posterior (bool): If true, will return fixed posterior
                values instead model calculated values.

        Returns:

        """
        self.model = copy.deepcopy(model)
        self.n_top = n_top
        self.in_eval = False
        self.break_posterior = break_posterior

    def __call__(self, x):
        """Calls the model on input x and returns the reduced (n_top) output
        
        Args: 
            x (torch.tensor):  Same as any model input
        
        Returns: 
            (torch.tensor): Returns (n_top,) dimensional torch.tensor with
                scores on top classes.
        """
        output = self.model(x)
        if self.in_eval is False:
            return output

        reduced = torch.zeros(output.shape)
        arr = output.detach().cpu().numpy()
        to_del = arr.argsort(axis=1)[:, -self.n_top:]
        
        for idx, img in enumerate(to_del):
            for idy, label in enumerate(img[::-1]):
                if self.break_posterior:
                    reduced[idx][label] = 1./(idy+1)
                else:
                    reduced[idx][label] = output[idx][label]
        return reduced

    def eval(self):
        """Sets the model and wrapper to eval mode
        """
        self.in_eval = True
        self.model.eval()

    def train(self):
        """Sets the model and wrapped to train mode
        """
        self.in_eval = False
        self.model.train()


def distill_model(teacher=None, student=None, data_loader=None,
                  test_loader=None, optimizer=None,
                  criterion=softCrossEntropy(), n_epochs=0, T=1.,
                  verbose=False):
    """Performs defensive distillation at desired temperature
    
    Args: 
        teacher (nn.Module): Teacher model used to in distillation.
        student (nn.Module): Student model into which to distill. If left as
            None will copy and randomly initialize the teacher.
        data_loader (Dataloader): Dataloader pointing to training dataset.
        test_loader (Dataloader): Dataloader poinitng to validation dataset.
        optimizer (nn.optim): Optimizer for distillation.
        criterion (nn.Module): Criterion for loss calculation. Default is
            softCrossEntropy(alpha = 0.95)
        n_epochs (int): Maximum number of epochs during distillation.
        T (int): Distillation temperature.  Assumes the teacher was trained at
            the same temperature.
        verbose (bool): If True will output loss at each training step.
    """
