from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)
