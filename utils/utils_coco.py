from torchvision.datasets import CocoDetection
from torch.utils.data._utils.collate import default_collate


class CocoWithIndex(CocoDetection):
    """
    Dataset COCO qui renvoie (image, annotations, idx).
    """
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # (image, list_of_dict)
        return img, target, idx

def coco_collate_fn_keep_target_with_index(batch):
    """
    batch[i] = (img, annots, idx)
    => On veut renvoyer (images, annots, idxs).
    """
    imgs = [sample[0] for sample in batch]
    annots = [sample[1] for sample in batch]
    idxs = [sample[2] for sample in batch]

    images_collated = default_collate(imgs)   # Tensor [B, 3, H, W]
    annots_collated = annots                 # liste python
    idxs_collated = default_collate(idxs)    # Tensor [B]

    return images_collated, annots_collated, idxs_collated


def build_cat_name_to_id_map(coco_obj):
    """
    Construit un dict { "person": 1, "bicycle": 2, ... }
    à partir de l'objet COCO (ex: train_dataset.coco).
    """
    cat_ids = coco_obj.getCatIds()
    cats_info = coco_obj.loadCats(cat_ids)
    name_to_id = {}
    for c in cats_info:
        name_to_id[c['name']] = c['id']
    return name_to_id

def parse_objectives_coco(arg_str: str):
    """
    Parse une chaîne de la forme 'person:1,skateboard:2,motorcycle:2'
    en un dict Python: {'person': 1.0, 'skateboard': 2.0, 'motorcycle': 2.0}.
    """
    if not arg_str:
        return {}
    items = arg_str.split(",")
    d = {}
    for item in items:
        name_val = item.split(":")
        cat_name = name_val[0].strip()
        cat_reward = float(name_val[1])
        d[cat_name] = cat_reward
    return d