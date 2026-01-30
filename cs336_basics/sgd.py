from collections.abc import Callable
from typing import Optional
import torch
import math
import matplotlib.pyplot as plt

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad

                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1
        return loss
    

def run_experiment(learning_rates, iterations=10):
    results = {}

    torch.manual_seed(42)
    initial_weights = 5 * torch.randn((10, 10))

    for lr in learning_rates:
        weights = torch.nn.Parameter(initial_weights.clone())
        opt = SGD([weights], lr=lr)
        loss_history = []

        print(f"\n--- Testing Learning Rate: {lr} ---")
        for t in range(iterations):
            opt.zero_grad()
            loss = (weights**2).mean()
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            print(f"Iteration {t}: Loss = {current_loss:.4f}")
            
            loss.backward()
            opt.step()
            
        results[lr] = loss_history
    
    return results

def plot_results(results):
    plt.figure(figsize=(10, 6))
    for lr, losses in results.items():
        plt.plot(range(len(losses)), losses, marker='o', label=f'LR = {lr}')
    
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Log Scale)')
    plt.title('SGD Learning Rate Tuning: Loss vs. Iterations')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    lrs_to_test = [1e0, 1e1, 1e2, 1e3]
    
    experiment_data = run_experiment(lrs_to_test, iterations=10)

    plot_results(experiment_data)