import numpy as np
from collections import defaultdict
import torch
import random

def filter_dataset_all_classes(dataset, num_per_class=50):
    class_to_indices = defaultdict(list)
    
    for idx, label in enumerate(dataset.targets):
        class_to_indices[label].append(idx)
    
    selected_indices = []
    for label in range(len(dataset.classes)):
        selected_indices.extend(class_to_indices[label][:num_per_class])
    
    selected_indices = np.array(selected_indices)
    
    dataset.data = dataset.data[selected_indices]
    dataset.targets = [dataset.targets[i] for i in selected_indices]
    
    return dataset

def filter_mnist_by_class(dataset, num_per_class=50, random_subset=True):
    if isinstance(dataset.targets, torch.Tensor):
        targets = dataset.targets
    else:
        targets = torch.tensor(dataset.targets)
    
    classes = torch.unique(targets)
    selected_indices = []

    for cls in classes:
        cls_indices = (targets == cls).nonzero(as_tuple=True)[0]
        if len(cls_indices) < num_per_class:
            raise ValueError(f"Classe {cls.item()} a seulement {len(cls_indices)} images, moins que le nombre demandé {num_per_class}.")
        
        if random_subset:
            permuted_indices = cls_indices[torch.randperm(len(cls_indices))]
            sampled = permuted_indices[:num_per_class]
        else:
            sampled = cls_indices[:num_per_class]
        
        selected_indices.append(sampled)
    
    selected_indices = torch.cat(selected_indices)
    
    dataset.data = dataset.data[selected_indices]
    dataset.targets = dataset.targets[selected_indices]

    print(f"Dataset filtré: {len(dataset)} images ({num_per_class} par classe).")

# def filter_coco_inplace(coco_dataset, num=5000):
#     if num >= len(coco_dataset.ids):
#         print(f"Le dataset a déjà {len(coco_dataset.ids)} images, pas besoin de filtrer.")
#         return
    
#     new_ids = random.sample(coco_dataset.ids, k=num)
#     coco_dataset.ids = new_ids
#     print(f"COCO dataset filtré à {num} images.")

from collections import defaultdict

def filter_coco_inplace(coco_dataset, target_classes, retaining_ratio=0.5):
    """
    Filtre le dataset COCO pour conserver une distribution réaliste d'images.

    Args:
      coco_dataset: Objet COCODetection avec la liste des IDs dans `coco_dataset.ids`.
      target_classes: Set ou liste des noms d'objets cibles, ex: {"person", "car", "bus", ...}.
      retaining_ratio: Ratio global d'images à conserver après filtrage (ex: 0.5 pour 50%).

    Retourne:
      Le dataset filtré (avec coco_dataset.ids mis à jour).
    """
    cat_id_to_name = {}
    for cat in coco_dataset.coco.loadCats(coco_dataset.coco.getCatIds()):
        cat_id_to_name[cat['id']] = cat['name']
    
    images_with_target = []
    images_without_target = []
    target_counts = {cls: 0 for cls in target_classes}

    for img_id in coco_dataset.ids:
        ann_ids = coco_dataset.coco.getAnnIds(imgIds=img_id)
        ann_list = coco_dataset.coco.loadAnns(ann_ids)
        
        found_target = False
        image_targets = set()
        for ann in ann_list:
            cat_id = ann.get("category_id")
            if cat_id is not None:
                cat_name = cat_id_to_name.get(cat_id, None)
                if cat_name in target_classes:
                    found_target = True
                    image_targets.add(cat_name)
        if found_target:
            images_with_target.append(img_id)
            for t in image_targets:
                target_counts[t] += 1
        else:
            images_without_target.append(img_id)
    
    total_with = len(images_with_target)
    total_without = len(images_without_target)
    print(f"Total images with target objects: {total_with}")
    for cls, count in target_counts.items():
        print(f"  {cls}: {count} images")
    print(f"Total images without target objects: {total_without}")
    
    n_total = int(len(coco_dataset.ids) * retaining_ratio)
    # actual config, keep 80% of target images 20% non target
    n_with = int(n_total * 0.8)
    n_without = n_total - n_with

    new_ids = random.sample(images_with_target, min(n_with, len(images_with_target))) + \
              random.sample(images_without_target, min(n_without, len(images_without_target)))
    
    coco_dataset.ids = new_ids
    print(f"Dataset final filtré: {len(coco_dataset.ids)} images.")
    return coco_dataset