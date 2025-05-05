import torch
from utils.device import DEVICE

# Autre logique possible si on veut la reward maximale dès que l'objet max satisfait la conditoin, par ex. un chien a lunette, c'est pas un humain, ni un homme mais il a des lunettes donc il prendra +3
# Au lieu de r_i += ...
# found_person = False
# found_male = False
# found_glasses = False
# for ann in ann_list:
#     cat_id = ann["category_id"]
#     cat_name = id_to_name[cat_id]
#     if cat_name == "person":
#         found_person = True
    
#     if "attributes" in ann:
#         if ann["attributes"].get("male", False):
#             found_male = True
#         if ann["attributes"].get("glasses", False):
#             found_glasses = True
# 
# if found_person: r_i += objectives["person"]
# if found_male: r_i += objectives["male"]
# if found_glasses: r_i += objectives["glasses"]

# def reward_function_coco(
#     selected_indices,      # [batch_size, subset_size]
#     all_annotations,       # liste d'annotations pour chaque image
#     cat_name_to_id_map,    # ex: {"person":1, "skateboard":24, "motorcycle":4, ...}
#     objectives,            # ex: {"person":1.0, "skateboard":2.0, "motorcycle":3.0}
#     device="cpu"
# ):
#     """
#     Calcule un reward pour chaque séquence.
    
#     Règle:
#       - "person" seule => + objectives["person"]
#       - S'il y a "person" dans l'image, alors on ajoute:
#           + objectives["skateboard"] si skateboard présent
#           + objectives["motorcycle"] si motorcycle présent
#       => skateboard ou motorcycle seuls => 0

#     """
#     batch_size, subset_size = selected_indices.shape
#     rewards = torch.zeros(batch_size, device=device)

#     # Map inverse cat_id -> cat_name
#     id_to_name = {v: k for k, v in cat_name_to_id_map.items()}

#     for i in range(batch_size):
#         row_indices = selected_indices[i].tolist()
#         total_reward_for_seq = 0.0

#         # Parcourt toutes les images de la séquence
#         for img_idx in row_indices:
#             ann_list = all_annotations[img_idx]  # liste d'objets dans cette image

#             # Flags
#             person_found = False
#             skateboard_found = False
#             motorcycle_found = False

#             for ann in ann_list:
#                 cat_id = ann.get("category_id", None)
#                 if cat_id is None:
#                     continue
#                 cat_name = id_to_name.get(cat_id, None)

#                 if cat_name == "person":
#                     person_found = True
#                 elif cat_name == "skateboard":
#                     skateboard_found = True
#                 elif cat_name == "motorcycle":
#                     motorcycle_found = True

           
#             image_reward = 0.0
#             if person_found:
#                 image_reward += objectives.get("person", 0.0)
#                 if skateboard_found:
#                     image_reward += objectives.get("skateboard", 0.0)
#                 if motorcycle_found:
#                     image_reward += objectives.get("motorcycle", 0.0)

#             total_reward_for_seq += image_reward

#         rewards[i] = total_reward_for_seq

#     return rewards

# def reward_function_multiclass(selected_indices, all_labels, target_classes, class_indices):
#     batch_size, subset_size = selected_indices.shape
#     rewards = torch.zeros(batch_size).to(all_labels.device)
    
#     for i in range(batch_size):
#         indices = selected_indices[i]
#         selected_labels = all_labels[indices]
#         reward = 0.0
#         for class_name, desired_count in target_classes.items():
#             actual_count = (selected_labels == class_indices[class_name]).sum().float()
#             reward += (actual_count / desired_count)
#             # if actual_count == desired_count:
#             #     reward += 1.0
#         rewards[i] = reward / len(target_classes)
    
#     return rewards

def reward_function_multiclass(
    selected_indices,      
    all_labels,            
    objectives_cifar,      
    class_indices,         
):
    batch_size, subset_size = selected_indices.shape
    device = all_labels.device
    rewards = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        row_indices = selected_indices[i]
        selected_lbls = all_labels[row_indices]
        total_reward_for_seq = 0.0
        
        for class_name, reward_val in objectives_cifar.items():
            actual_count = (selected_lbls == class_indices[class_name]).sum().item()
            total_reward_for_seq += actual_count * reward_val

        rewards[i] = total_reward_for_seq
    return rewards

def reward_function_coco(
    selected_indices,      
    all_annotations,       
    cat_name_to_id_map,    
    objectives,            
    device="cpu"
):
  
    batch_size, subset_size = selected_indices.shape
    rewards = torch.zeros(batch_size, device=device)

    id_to_name = {v: k for k, v in cat_name_to_id_map.items()}

    for i in range(batch_size):
        row_indices = selected_indices[i].tolist()
        total_reward_for_seq = 0.0

        for img_idx in row_indices:
            ann_list = all_annotations[img_idx]  

            categories_in_this_image = set()
            for ann in ann_list:
                cat_id = ann.get("category_id", None)
                if cat_id is not None:
                    cat_name = id_to_name.get(cat_id, None)
                    if cat_name is not None:
                        categories_in_this_image.add(cat_name)

            image_reward = 0.0
            for cat_name in categories_in_this_image:
                if cat_name in objectives:
                    image_reward += objectives[cat_name]

            total_reward_for_seq += image_reward

        rewards[i] = total_reward_for_seq

    return rewards

def build_reward_fn(cfg, objectives, cat_map, class_indices):
    if cfg.dataset == "COCO":
        def _fn(indices, all_annots, *_):
            return reward_function_coco(indices, all_annots, cat_map, objectives, device=DEVICE)
        return _fn
    return reward_function_multiclass