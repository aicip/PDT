import torch
import numpy as np

class SingleClassBatchSampler(torch.utils.data.Sampler):
    """Only sample one class in each batch."""
    def __init__(self, dataset, batch_size=256):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))

        # Group sample indices by class
        self.class_indices = {}
        for idx in range(len(dataset)):
            label = dataset.targets[idx]
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        # Compute total number of batches
        self.num_batches = sum(len(indices) // batch_size for indices in self.class_indices.values())
    
    def __iter__(self):
        # Copy indices to avoid modifying the original data
        remaining_indices = {
            k: v.copy() for k, v in self.class_indices.items()
        }

        # Store available classes
        available_classes = list(remaining_indices.keys())
        
        while len(available_classes) > 0:
            # Randomly select a class
            class_id = np.random.choice(available_classes)
            class_indices = remaining_indices[class_id]
            
            if len(class_indices) >= self.batch_size:
                # randomly sample a batch of indices from the selected class
                batch_indices = np.random.choice(
                    class_indices, 
                    size=self.batch_size, 
                    replace=False
                ).tolist()

                # Update remaining indices
                remaining_indices[class_id] = list(
                    set(class_indices) - set(batch_indices)
                )
                
                yield batch_indices

            # If there are not enough samples of this class for a batch, remove from available_classes
            if len(remaining_indices[class_id]) < self.batch_size:
                available_classes.remove(class_id)
    
    def __len__(self):
        return self.num_batches