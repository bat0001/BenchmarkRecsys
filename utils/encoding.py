import torch
from torch.utils.data import DataLoader
from utils.utils_coco import coco_collate_fn_keep_target_with_index
from utils.device import DEVICE

def encode_dataset(backbone, ds, cfg):
    kwargs = {}
    is_coco = cfg.dataset == "COCO"
    if is_coco:
        kwargs["collate_fn"] = coco_collate_fn_keep_target_with_index
    dl = DataLoader(ds, batch_size=128, shuffle=False, **kwargs)

    embs, metas = [], []
    for batch in dl:
        if is_coco:
            imgs, anns, _ = batch
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                z = backbone.encode(imgs)[0] if hasattr(backbone, "encode") else backbone(imgs)
            embs.append(z.cpu()); metas.extend(anns)
        else:
            imgs, lbls = batch
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                z = backbone.encode(imgs)[0] if hasattr(backbone, "encode") else backbone(imgs)
            embs.append(z.cpu()); metas.append(lbls)
    embs = torch.cat(embs, dim=0)
    if is_coco:
        return embs, metas
    return embs, torch.cat(metas, dim=0)