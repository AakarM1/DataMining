import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FederatedClient:
    def __init__(self, model, dataset, heterogeneity_factor):
        self.model = model
        self.dataset = dataset
        self.heterogeneity_factor = heterogeneity_factor


    def compute_update(self, global_weights):
        try:
            self.model.set_weights(global_weights)

            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            cmcr_loss_value = 0.0
            for data, target in self.dataset:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)

                # Cross-Modal Consistency Loss (use dummy MRI/CT inputs for now)
                if hasattr(self.model, "compute_cmcr_loss"):
                    x_ct = torch.randn_like(data)  # Placeholder for CT data
                    x_mri = torch.randn_like(data)  # Placeholder for MRI data
                    cmcr_loss = self.model.compute_cmcr_loss(x_ct, x_mri)
                    loss += cmcr_loss
                    cmcr_loss_value += cmcr_loss.item()

                loss.backward()
                optimizer.step()

            return self.model.get_weights(), 1 / (1 + cmcr_loss_value)
        except Exception as e:
            print(f"[ERROR] Client compute_update with CMCR failed: {str(e)}")
            raise