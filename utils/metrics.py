def accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    return correct / len(labels)
