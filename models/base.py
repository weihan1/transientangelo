import torch
import torch.nn as nn

from utils.misc import get_rank

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.setup()
        if self.config.get('weights', None):
            self.load_state_dict(torch.load(self.config.weights))
        # self.attach_gradient_hooks()
        
    # def attach_gradient_hooks(self):
    #     # Attach hooks to all applicable layers
    #     for name, module in self.named_modules():
    #         module.register_backward_hook(self.print_gradients)

    # def print_gradients(self, module, grad_input, grad_output):
    #     print(f"Gradients on {module.__class__.__name__}: {grad_output[0]}")   
        
    def setup(self):
        raise NotImplementedError
    
    def update_step(self, epoch, global_step):
        pass
    
    def train(self, mode=True):
        return super().train(mode=mode)
    
    def eval(self):
        return super().eval()
    
    def regularizations(self, out):
        return {}
    
    @torch.no_grad()
    def export(self, export_config):
        return {}
