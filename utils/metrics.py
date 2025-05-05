import math
import torch
from collections import Counter

from utils.rewards import reward_function_multiclass, reward_function_coco
from utils.sampling import sample_images

def precision_score(true_labels, predicted_indices, target_classes, class_indices, device):
    batch_size, subset_size = predicted_indices.shape
    true_positives = 0
    false_positives = 0
    
    target_class_indices_set = set(class_indices.values())
    
    for i in range(batch_size):
        indices = predicted_indices[i]
        selected_labels = true_labels[indices]
        correct = (selected_labels.unsqueeze(1) == torch.tensor(list(target_class_indices_set), device=device)).any(dim=1).sum().item()
        true_positives += correct
        false_positives += (subset_size - correct)
            
    precision = true_positives / (true_positives + false_positives + 1e-8)
    print(f"Total True Positives: {true_positives}, Total False Positives: {false_positives}, Precision: {precision:.4f}")
    return precision

def recall_score_func(true_labels, predicted_indices, target_classes, class_indices, device, total_target_images=None):
    if total_target_images is None:
        target_class_indices_set = set(class_indices.values())
        total_target_images = sum([(true_labels == idx).sum().item() for idx in target_class_indices_set])
        print(f"Total Target Images: {total_target_images}")
    
    selected_labels = true_labels[predicted_indices].flatten()
    target_class_values = list(class_indices.values())
    
    if hasattr(torch, 'isin'):
        target_class_tensor = torch.tensor(target_class_values, device=device)
        mask = torch.isin(selected_labels, target_class_tensor)
    else:
        mask = torch.zeros_like(selected_labels, dtype=torch.bool, device=device)
        for val in target_class_values:
            mask = mask | (selected_labels == val)
    
    selected_labels = selected_labels[mask]
    
    selected_counts = torch.bincount(selected_labels, minlength=max(target_class_values)+1)
    print(f"Selected Counts per Class: {selected_counts.tolist()}")
    
    target_class_images = {cls: (true_labels == idx).sum().item() for cls, idx in class_indices.items()}
    print(f"Target Class Images: {target_class_images}")
    
    selected_correct = 0
    for cls, count in target_class_images.items():
        cls_idx = class_indices[cls]
        selected_correct += min(selected_counts[cls_idx].item(), count)
        print(f"Class {cls}: Selected Correct = {min(selected_counts[cls_idx].item(), count)}, Total = {count}")
    
    recall = selected_correct / (total_target_images + 1e-8)
    print(f"Selected Correct: {selected_correct}, Recall: {recall:.4f}")
    return recall

def f1_score_func(precision, recall):
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    print(f"F1-Score: {f1:.4f}")
    return f1

def exact_match_ratio(true_labels, predicted_indices, target_classes, class_indices):
    batch_size, subset_size = predicted_indices.shape
    correct = 0
    for i in range(batch_size):
        indices = predicted_indices[i]
        selected_labels = true_labels[indices]
        match = True
        for class_name, desired_count in target_classes.items():
            actual_count = (selected_labels == class_indices[class_name]).sum().item()
            if actual_count != desired_count:
                match = False
                break
        if match:
            correct += 1
    exact = correct / batch_size
    print(f"Exact Match Ratio: {exact:.4f}")
    return exact

def evaluate_model(model, embeddings, labels, target_classes, class_indices, subset_size, device, top_k=True):
    model.eval()
    selected_indices = []
    
    with torch.no_grad():
        if top_k:
            logits = model(embeddings.to(device)) 
            _, selected = torch.topk(logits, subset_size, dim=1)
            selected_indices = selected.cpu()
        else:
            all_probs = []
            batch_size_eval = 1024  
            num_batches = math.ceil(len(embeddings) / batch_size_eval)
            
            for i in range(num_batches):
                start = i * batch_size_eval
                end = min((i + 1) * batch_size_eval, len(embeddings))
                batch_emb = embeddings[start:end].to(device)
                logits = model(batch_emb) 
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                
                if i < 2:
                    print(f"Batch {i+1}/{num_batches} - Mean Prob: {probs.mean().item():.4f}, Std Prob: {probs.std().item():.4f}")
            
            all_probs = torch.cat(all_probs, dim=0)
            selected = torch.multinomial(all_probs, subset_size, replacement=False)
            selected_indices = selected
    
    precision = precision_score(labels, selected_indices, target_classes, class_indices, device)
    recall = recall_score_func(labels, selected_indices, target_classes, class_indices, device)
    f1 = f1_score_func(precision, recall)
    exact = exact_match_ratio(labels, selected_indices, target_classes, class_indices)
    
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Exact Match Ratio": exact
    }
    
    return metrics, selected_indices


def measure_sequence_diversity(policy, sample_fn, num_samples=10000, T=3):
    """
    Mesure la diversité des séquences générées par un policy, 
    en calculant l'entropie de Shannon sur la distribution empirique des séquences.
    """
    freq_counter = Counter()
    
    for _ in range(num_samples):
        seq, _ = sample_fn(policy, T)
        seq_tuple = tuple(seq)  
        freq_counter[seq_tuple] += 1
    
    total = sum(freq_counter.values()) 
    entropy = 0.0
    
    for seq_tuple, count in freq_counter.items():
        p = count / total
        entropy -= p * math.log(p, 2)
    return entropy


def measure_image_diversity(policy, sample_fn, num_samples=10000, T=3):
    """
    Mesure la diversité au niveau des images sélectionnées,
    en calculant l'entropie de Shannon sur la distribution des indices d'images.
    """
    freq_counter = Counter()
    total_selected = 0
    
    for _ in range(num_samples):
        seq, _ = sample_fn(policy, T)
        for img_idx in seq:
            freq_counter[img_idx] += 1
            total_selected += 1
    
    entropy = 0.0
    for img_idx, count in freq_counter.items():
        p = count / total_selected
        entropy -= p * math.log(p, 2)
    
    return entropy



def measure_sequence_diversity_from_list(list_of_seqs):
    freq_counter = Counter()
    total = len(list_of_seqs)

    for seq in list_of_seqs:
        seq_tuple = tuple(seq)
        freq_counter[seq_tuple] += 1

    entropy = 0.0
    for _, count in freq_counter.items():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def measure_image_diversity_from_list(list_of_seqs):
    freq_counter = Counter()
    total_images = 0

    for seq in list_of_seqs:
        for img_idx in seq:
            freq_counter[img_idx] += 1
            total_images += 1
    
    entropy = 0.0
    for _, count in freq_counter.items():
        p = count / total_images
        entropy -= p * math.log2(p)
    return entropy

def measure_distinct_relevant_sequences(
    policy, 
    embeddings, 
    labels, 
    target_classes, 
    class_indices, 
    device,
    subset_size=3, 
    num_samples=5000
):

    policy.eval()
    distinct_relevant = set()
    
    for _ in range(num_samples):
        seq = sample_images(policy, embeddings, subset_size=subset_size, top_k=False, device=device)
        
        seq_tensor = torch.tensor(seq, device=device).unsqueeze(0) 
        r = reward_function_multiclass(seq_tensor, labels, target_classes, class_indices)  

        if r.item() > 0:
            seq_sorted = tuple(sorted(seq))
            distinct_relevant.add(seq_sorted)
    
    return len(distinct_relevant)


def image_contains_target(ann_list, target_cat_ids):
    for ann in ann_list:
        if ann["category_id"] in target_cat_ids:
            return True
    return False

def recall_score_coco(all_annotations, selected_indices, target_classes, cat_name_to_id_map):
    target_cat_ids = set()
    for class_name in target_classes.keys():
        if class_name in cat_name_to_id_map:
            target_cat_ids.add(cat_name_to_id_map[class_name])

    N = len(all_annotations)
    positive_indices = []
    for i in range(N):
        ann_list = all_annotations[i]
        if image_contains_target(ann_list, target_cat_ids):
            positive_indices.append(i)

    total_positives = len(positive_indices)

    selected_set = set()
    batch_size, subset_size = selected_indices.shape
    for i in range(batch_size):
        row_idxs = selected_indices[i].tolist()
        for idx in row_idxs:
            selected_set.add(idx)

    tp = sum(1 for i in positive_indices if i in selected_set)

    recall = tp / (total_positives + 1e-8)
    return recall

def precision_score_coco(all_annotations, selected_indices, target_classes, cat_name_to_id_map):
    target_cat_ids = set()
    for class_name in target_classes.keys():
        if class_name in cat_name_to_id_map:
            target_cat_ids.add(cat_name_to_id_map[class_name])

    tp = 0
    fp = 0

    batch_size, subset_size = selected_indices.shape
    for i in range(batch_size):
        row_idxs = selected_indices[i].tolist()  

        for img_idx in row_idxs:
            ann_list = all_annotations[img_idx]
            if image_contains_target(ann_list, target_cat_ids):
                tp += 1
            else:
                fp += 1

    precision = tp / (tp + fp + 1e-8)
    return precision

def f1_score_coco(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# def evaluate_model_coco(
#     model, 
#     all_embeddings, 
#     all_annotations, 
#     target_classes, 
#     cat_name_to_id_map,
#     subset_size,
#     device,
#     top_k=True
# ):

#     model.eval()

#     with torch.no_grad():
#         logits = model(all_embeddings.to(device)) 
#         if top_k:
#             _, selected_indices = torch.topk(logits, subset_size, dim=1)
#         else:
#             probs = torch.softmax(logits, dim=1)
#             selected_indices = torch.multinomial(probs, subset_size, replacement=False)
#         selected_indices = selected_indices.cpu()

#     precision = precision_score_coco(all_annotations, selected_indices, target_classes, cat_name_to_id_map)
#     recall = recall_score_coco(all_annotations, selected_indices, target_classes, cat_name_to_id_map)
#     f1 = f1_score_coco(precision, recall)
    
#     exact_match = 0.0
    
#     metrics = {
#         "Precision": precision,
#         "Recall": recall,
#         "F1-Score": f1,
#         "Exact Match Ratio": exact_match
#     }
#     return metrics, selected_indices

def evaluate_model_coco(
    model, 
    all_embeddings, 
    all_annotations,  
    objectives,       
    cat_name_to_id_map,
    subset_size,
    device,
    top_k=False
):
    
    print(f'subset size: {subset_size}')
    print(f'top k = {top_k}')
    model.eval()
    with torch.no_grad():
        logits = model(all_embeddings.to(device))
        print("Logits mean:", logits.mean().item(), "std:", logits.std().item())
        if top_k:
            _, selected_indices = torch.topk(logits, subset_size, dim=1)
        else:
            # temp = 10.0  # Par exemple, à ajuster
            # probs = torch.softmax(logits / temp, dim=1)
            probs = torch.softmax(logits, dim=1)
            
            selected_indices = torch.multinomial(probs, subset_size, replacement=False)
        selected_indices = selected_indices.cpu()
    
    rewards = reward_function_coco(
        selected_indices=selected_indices,
        all_annotations=all_annotations,
        cat_name_to_id_map=cat_name_to_id_map,
        objectives=objectives,
        device=device
    )
    
    mean_reward = rewards.mean().item()
    success_rate = (rewards > 0).float().mean().item()
    
    metrics = {
        "Mean Reward": mean_reward,
        "Success Rate": success_rate
    }
    return metrics, selected_indices
