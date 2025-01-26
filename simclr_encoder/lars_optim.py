from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    Args:
        params (iterable): Parameters to optimize or dictionaries defining parameter groups.
        lr (float): Learning rate.
        momentum (float): Momentum factor (default: 0).
        weight_decay (float): Weight decay (L2 penalty) (default: 0).
        eps (float): Epsilon for numerical stability (default: 1e-8).
        trust_coef (float): Trust coefficient for adaptive learning rate (default: 0.001).
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, eps=1e-8, trust_coef=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, trust_coef=trust_coef)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss (optional): The loss if the closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            trust_coef = group["trust_coef"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(grad)
                adaptive_lr = trust_coef * param_norm / (grad_norm + eps) if param_norm > 0 and grad_norm > 0 else 1.0

                scaled_lr = adaptive_lr * lr
                state = self.state[p]

                # Initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.clone(grad).detach()
                else:
                    state["momentum_buffer"].mul_(momentum).add_(grad)

                p.data.add_(state["momentum_buffer"], alpha=-scaled_lr)

        return loss