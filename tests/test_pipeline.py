import unittest
from models.unet import UNet
from data.loaders import get_dataloader
from fedseg.federated import FederatedServer
from fedseg.client import FederatedClient

class TestFederatedPipeline(unittest.TestCase):
    def test_model_initialization(self):
        model = UNet()
        self.assertIsNotNone(model)

    def test_dataloader(self):
        dataloader = get_dataloader(batch_size=4)
        for data, target in dataloader:
            self.assertEqual(data.shape[0], 4)
            self.assertEqual(target.shape[0], 4)

    def test_federated_server(self):
        model = UNet()
        server = FederatedServer(model, learning_rate=0.01)
        self.assertIsNotNone(server.global_weights)

if __name__ == "__main__":
    unittest.main()
