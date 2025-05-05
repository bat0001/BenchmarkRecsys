from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch.utils.data._utils.collate import default_collate

from utils.filter import filter_mnist_by_class


class CocoWithIndex(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return img, target, idx
    

def coco_collate_fn_keep_target(batch):    
    images = [item[0] for item in batch]
    annots = [item[1] for item in batch]  
    images_collated = default_collate(images)     
    return images_collated, annots

def coco_collate_fn_keep_target_with_index(batch):
    imgs = [sample[0] for sample in batch]
    annots = [sample[1] for sample in batch]
    idxs = [sample[2] for sample in batch]

    images_collated = default_collate(imgs)
    annots_collated = annots               
    idxs_collated = default_collate(idxs)  

    return images_collated, annots_collated, idxs_collated

def prepare_dataset(dataset_name="CIFAR-10", root_coco="./coco/train2017", annFile_coco="./coco/annotations/instances_train2017.json"):
   
    if dataset_name in ["CIFAR-10", "CIFAR-100"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif dataset_name in ["MNIST", "FashionMNIST"]:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif dataset_name == "COCO":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Dataset {dataset_name} unsupported.")
    
    if dataset_name == "CIFAR-10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "CIFAR-100":
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "COCO":
        train_dataset = CocoWithIndex(
            root="data/coco/images/train2017/",
            annFile="data/coco/annotations/instances_train2017.json",
            transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} unsupported.")
    
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True,
                collate_fn=coco_collate_fn_keep_target
    )
    
    return train_dataset, train_loader