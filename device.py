import torch


class DeviceManager:

    def __init__(self, device):
        self.device = device
    def set_device(self):

        if self.device:
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(self, data,device):
        'we move our tensor to given device'
        if isinstance(data, (list,tuple)):
            return [to_device(x,device) for x in data]

        return data.to(device,non_blocking = True)
