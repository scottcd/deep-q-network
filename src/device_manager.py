import torch

class DeviceManager:
    __instance = None

    @staticmethod
    def get_instance():
        """Static method to retrieve the singleton instance."""
        if DeviceManager.__instance is None:
            DeviceManager()
        return DeviceManager.__instance

    def __init__(self):
        """Private constructor to initialize the singleton instance."""
        if DeviceManager.__instance is not None:
            raise Exception("DeviceManager is a singleton class. Use get_instance() to retrieve the instance.")
        else:
            DeviceManager.__instance = self
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        """Method to retrieve the current device."""
        return self.device