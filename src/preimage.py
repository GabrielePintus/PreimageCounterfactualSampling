import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict
from tqdm import tqdm







class WrappedModel(nn.Module):
    """
    A wrapper for a neural network model that enforces linear constraints
    on the output logits via a fixed linear layer.

    The wrapper constructs a constraint matrix `C` and bias `b` such that 
    the logit of a specific class (the chosen label) is compared against 
    all other classes.

    Attributes
    ----------
    model : nn.Module
        The base model whose outputs are constrained.
    label : int
        The target class label for which constraints are applied.
    device : torch.device
        The device on which computations are performed.
    n_labels : int
        The total number of labels/classes.
    constraint_layer : nn.Linear
        A fixed linear layer implementing the constraints.
    """

    def __init__(self, model: nn.Module, label: int, device: torch.device, n_labels: int = 10):
        """
        Initialize the wrapped model.

        Parameters
        ----------
        model : nn.Module
            The base neural network model.
        label : int
            The class label to constrain against others.
        device : torch.device
            The device (CPU/GPU) to place tensors on.
        n_labels : int, optional (default=10)
            Total number of labels/classes.
        """
        super().__init__()
        self.model = model
        self.label = label
        self.device = device
        self.n_labels = n_labels

        self.constraint_layer = nn.Linear(
            in_features=n_labels,
            out_features=n_labels - 1,
            bias=True
        )
        self._init_weights()

    @staticmethod
    def build_C_matrix(label: int, n_labels: int = 10) -> torch.Tensor:
        """
        Construct the constraint matrix C.

        Parameters
        ----------
        label : int
            The chosen class label.
        n_labels : int, optional (default=10)
            The number of labels/classes.

        Returns
        -------
        torch.Tensor
            The constraint matrix of shape (n_labels-1, n_labels).
        """
        C = -torch.eye(n_labels - 1)
        C = torch.cat((C[:, :label], torch.ones(n_labels - 1, 1), C[:, label:]), dim=1)
        return C

    @classmethod
    def build_C_b(cls, label: int, n_labels: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Construct the constraint matrix C and bias vector b.

        Parameters
        ----------
        label : int
            The chosen class label.
        n_labels : int, optional (default=10)
            The number of labels/classes.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            The constraint matrix C and the bias vector b.
        """
        C = cls.build_C_matrix(label, n_labels)
        b = torch.zeros(n_labels - 1)
        return C, b

    def _init_weights(self) -> None:
        """Initialize the fixed weights of the constraint layer."""
        C, b = self.build_C_b(self.label, self.n_labels)
        self.constraint_layer.weight = nn.Parameter(C, requires_grad=False)
        self.constraint_layer.bias = nn.Parameter(b, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the wrapped model with constraints.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            The constrained output of shape (batch_size, n_labels-1).
        """
        y = self.model(x)                     # Model logits
        g = self.constraint_layer(y)          # Apply constraints
        return g.view(x.size(0), -1)          # Flatten per batch






def get_lower_bound(A):
    lA = A['/x']['lA'].detach().cpu().numpy().squeeze()
    lbias = A['/x']['lbias'].detach().cpu().numpy().reshape(-1)
    return lA, lbias

def get_upper_bound(A):
    uA = A['/x']['uA'].detach().cpu().numpy().squeeze()
    ubias = A['/x']['ubias'].detach().cpu().numpy().reshape(-1)
    return uA, ubias




class PreimageApproximation:
    """
    Class to generate counterfactuals for a given dataset and model.
    """
    def __init__(self, model, dataset, cnn=False):
        """
        Initialize the PreimageApproximation class.

        Parameters:
        - model: The model to generate counterfactuals for.
        - dataset: The dataset to use for generating counterfactuals.
        """
        self.model = model
        self.dataset = dataset
        self.n_classes = len(self.dataset.tensors[1].unique())
        self.cnn = cnn

    @staticmethod
    def wrap_model(model, label, n_classes):
        wrapped_model = WrappedModel(model, label, n_classes)
        return wrapped_model
        
    def wrap(self, label):
        self.wrapped_model = PreimageApproximation.wrap_model(self.model, label, self.n_classes)
    
    def set_perturbation(self, eps=0.1, norm=2):
        self.ptb = PerturbationLpNorm(norm=norm, eps=eps)
    
    def bound_tensor(self, label): 
        X_selected = self.dataset.tensors[0][self.dataset.tensors[1] == label].float()
        if self.cnn:
            X_selected = X_selected.view(-1, 1, 28, 28)
        else:
            X_selected.view(-1, 28*28)
        X_selected_bounded = BoundedTensor(X_selected, self.ptb)
        
        self.X_selected = X_selected
        self.X_selected_bounded = X_selected_bounded
        
    def bound_model(self):
        if self.cnn:
            self.bounded_model = BoundedModule(self.wrapped_model, self.X_selected_bounded,
                                               bound_opts={"conv_mode": "patches"})
        else:
            self.bounded_model = BoundedModule(self.wrapped_model, self.X_selected_bounded)
        
    def setup(self, label, eps=0.1, norm=2):
        """
        Setup the counterfactual generation process.

        Parameters:
        - label: The label for which to generate counterfactuals.
        - eps: The perturbation size.
        - norm: The norm to use for perturbation.
        """
        self.wrap(label)
        self.set_perturbation(eps, norm)
        self.bound_tensor(label)
        self.bound_model()
        
    def compute_bounds(self):
        A = defaultdict(set)
        A[self.bounded_model.output_name[0]].add(self.bounded_model.input_name[0]) 
        
        _, _, A = self.bounded_model.compute_bounds(
            x=(self.X_selected_bounded, ), 
            # method="crown-optimized", 
            method='backward',
            return_A=True,
            needed_A_dict=A
        )
        
        key = next(iter(A.keys()))
        A = A[key]
        
        lA, lbias = get_lower_bound(A)
        uA, ubias = get_upper_bound(A)
        
        if self.cnn:
            lA = lA.reshape(-1, 1, 28, 28)
            uA = uA.reshape(-1, 1, 28, 28)
        else:
            lA = lA.reshape(-1, 28*28)
            uA = uA.reshape(-1, 28*28)
        
        lbias = lbias.reshape(-1)
        ubias = ubias.reshape(-1)
        
        return lA, lbias, uA, ubias
        
    def compute_all_bounds(self, save=False):
        """
        Compute the bounds for all classes.

        Returns:
        - A dictionary containing the lower and upper bounds for each class.
        """
        bounds = []
        for label in tqdm(range(self.n_classes)):
            self.setup(label)
            lA, lbias, uA, ubias = self.compute_bounds()
            bounds.append({
                'lower_bound': lA,
                'lower_bias': lbias,
                'upper_bound': uA,
                'upper_bias': ubias
            })
        if save:
            self.bounds = bounds
        return bounds






