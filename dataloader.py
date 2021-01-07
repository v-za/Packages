class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, manager):
        self.dl = dl
        self.manager = manager

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield manager.to_device(b)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
