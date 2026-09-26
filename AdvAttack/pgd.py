import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PGD:
    """
    Projected Gradient Descent (L-inf) adversarial attack.

    Operates on raw [0, 1] pixel-space inputs. The model passed in
    should be wrapped with NormalizedModel so that normalization
    happens inside forward().
    """

    def __init__(self, model, ep=8/255, epochs=10, step=2/255,
                 is_rand=True):
        self.model = model
        self.ep = ep
        self.epochs = epochs
        self.step = step
        self.is_rand = is_rand

    def generate_all(self, x, y):
        """
        Generate adversarial examples for ALL inputs (successful or not).

        Args:
            x: torch.Tensor (N, C, H, W), raw [0, 1] pixel values
            y: torch.Tensor (N,), ground-truth class indices

        Returns:
            (x_adv, y) — torch tensors, same shapes as inputs
        """
        self.model.eval()
        x = x.to(device)
        y = y.to(device)

        if self.is_rand:
            noise = torch.empty_like(x).uniform_(-self.ep, self.ep)
            x_adv = torch.clamp(x + noise, 0.0, 1.0)
        else:
            x_adv = x.clone()

        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = self.model(x_adv)
            loss = loss_fn(logits, y)
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + self.step * x_adv.grad.sign()
                # Project onto epsilon-ball around original
                x_adv = torch.max(torch.min(x_adv, x + self.ep),
                                  x - self.ep)
                # Clamp to valid pixel range
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        n_success = (self.model(x_adv.detach()).argmax(1) != y).sum().item()
        print(f"  PGD attack success: {n_success}/{len(x)} "
              f"({100 * n_success / len(x):.1f}%)")

        return x_adv.detach(), y