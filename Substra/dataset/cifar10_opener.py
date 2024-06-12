import os
import torch
import numpy as np
import substratools as tools

class CIFAR10Opener(tools.Opener):
    def get_data(self, folders):
        # The data is stored in a folder containing 'data.pt'
        data_files = [os.path.join(folder, "data.pt") for folder in folders]
        data = [torch.load(data_file) for data_file in data_files]
        
        images = torch.cat([d[0] for d in data], dim=0).numpy()
        labels = torch.cat([d[1] for d in data], dim=0).numpy()
        
        return {'images': images, 'labels': labels}

    def save_predictions(self, predictions, path):
        torch.save(predictions, path)

    def save_model(self, model, path):
        torch.save(model, path)

    def load_model(self, path):
        return torch.load(path)
    
    def fake_data(self, n_samples):
        # Generate fake data for testing purposes
        images = np.random.rand(n_samples, 3, 32, 32).astype(np.float32)
        labels = np.random.randint(0, 10, n_samples).astype(np.int64)
        return {'images': images, 'labels': labels}

if __name__ == "__main__":
    tools.execute(CIFAR10Opener())
