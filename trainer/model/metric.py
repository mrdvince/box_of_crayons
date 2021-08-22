import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=1):
    with torch.no_grad():
        _, top_class = torch.topk(output, k, dim=1)
        assert top_class.shape[0] == len(target)
        correct = 0
        for idx in range(k):
            correct += torch.sum(top_class[:, idx] == target).item()
        return correct / len(target)
