import torch


class DeviceManager:

    def __init__(self):
        pass

    def get_device(self):

        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(self, data, device):
        'we move our tensor to given device'
        if isinstance(data, (list,tuple)):
            return [self.to_device(x,device) for x in data]

        return data.to(device,non_blocking = True)
