import numpy as np
import torch


class FederatedServer:
    def __init__(self, model, learning_rate, momentum=0.9):
        self.global_weights = model.get_weights()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_term = None

    def aggregate_updates(self, client_updates, client_weights):
        """
        Perform weighted aggregation of client updates.
        """
        try:
            if len(client_updates) != len(client_weights):
                raise ValueError("Mismatch between number of updates and weights.")

            total_weight = sum(client_weights)
            if total_weight == 0:
                raise ValueError("Total client weight is zero, aggregation cannot proceed.")

            aggregated_update = None

            # Weighted aggregation
            for update, weight in zip(client_updates, client_weights):
                scaled_update = {
                    key: (weight / total_weight) * update[key]
                    for key in update.keys()
                }
                if aggregated_update is None:
                    aggregated_update = scaled_update
                else:
                    for key in aggregated_update.keys():
                        aggregated_update[key] += scaled_update[key]

            # Apply momentum
            if self.momentum_term is None:
                self.momentum_term = aggregated_update
            else:
                for key in self.momentum_term.keys():
                    self.momentum_term[key] = (
                        self.momentum * self.momentum_term[key] + aggregated_update[key]
                    )

            # Update global weights
            for key in self.global_weights.keys():
                self.global_weights[key] -= self.learning_rate * self.momentum_term[key]

            return self.global_weights
        except Exception as e:
            print(f"[ERROR] Server aggregation failed: {str(e)}")
            raise

