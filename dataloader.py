class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, manager,device):
        self.dl = dl
        self.manager = manager
        self.device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield self.manager.to_device(b,self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
