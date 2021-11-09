class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    # 定义model的某一层，需要根据hook提取feature和gradient
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compitability with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        # 对每个需要保存feature的层(target_layer)注册一个forward hook函数，用于提取forward过程中的feature
        # print("module ", module)
        print("save_activation ", input[0].shape, output.shape)
        activation = output # torch.Size([1, 2048, 34, 34])
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # 对每个需要保存gradient的层(target_layer)注册一个backward hook函数，用于提取backward过程中的gradient
        # Gradients are computed in reverse order
        # print("module ", module)
        print("save_gradient ", grad_input[0].shape, grad_output[0].shape)
        grad = grad_output[0] # [1, 2048, 34, 34]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x) # b, 3, h, w -> b, 1000

    def release(self):
        for handle in self.handles:
            handle.remove()
