import numpy as np
from utils.visualization import plot_umap_and_log

def collect_embeddings_cifar(
    policy,
    embeddings,
    all_labels,
    target_classes,   
    num_samples=500,
    subset_size=5,
    device="cpu"
):
    from utils.sampling import sample_images
    policy.eval()

    collected_points = []
    collected_scores = []

    class_indices_set = set(target_classes.values())

    for _ in range(num_samples):
        seq_indices = sample_images(policy, embeddings, subset_size=subset_size, top_k=False, device=device)
        
        seq_embeddings = [embeddings[idx].cpu().numpy() for idx in seq_indices]
        seq_mean = np.mean(seq_embeddings, axis=0)
        collected_points.append(seq_mean)

        seq_labels = [all_labels[idx].item() for idx in seq_indices]
        nb_targets = 0
        for lbl in seq_labels:
            if lbl in class_indices_set:
                nb_targets += 1

        collected_scores.append(nb_targets)

    points_arr = np.array(collected_points)  
    scores_arr = np.array(collected_scores)
    return points_arr, scores_arr

def collect_embeddings_from_policy(
    policy, 
    embeddings, 
    all_annotations, 
    cat_name_to_id_map, 
    num_samples=500,
    subset_size=5,
    device="cpu"
):
    from utils.sampling import sample_images
    policy.eval()

    collected_points = []  
    collected_scores = []  

    id_to_name = {v: k for k, v in cat_name_to_id_map.items()}

    for _ in range(num_samples):
        seq_indices = sample_images(
            policy, embeddings, subset_size=subset_size, 
            top_k=False, device=device
        )
        
        seq_embeddings = [embeddings[idx].cpu().numpy() for idx in seq_indices]
        seq_mean = np.mean(seq_embeddings, axis=0)
        collected_points.append(seq_mean)

        person_found = False
        skateboard_found = False
        motorcycle_found = False

        for img_idx in seq_indices:
            ann_list = all_annotations[img_idx]
            for ann in ann_list:
                cat_id = ann.get("category_id", None)
                cat_name = id_to_name.get(cat_id, "unknown")

                if cat_name == "person":
                    person_found = True
                elif cat_name == "skateboard":
                    skateboard_found = True
                elif cat_name == "motorcycle":
                    motorcycle_found = True
        
        score = compute_combination_score(person_found, skateboard_found, motorcycle_found)
        collected_scores.append(score)

    points_arr = np.array(collected_points)  
    scores_arr = np.array(collected_scores)  
    return points_arr, scores_arr


def compute_combination_score(person_found, skateboard_found, motorcycle_found):
    count = 0
    if person_found:
        count += 1
    if skateboard_found:
        count += 1
    if motorcycle_found:
        count += 1
    return count

def plot_umap_for_sequences(
    config,
    device,
    all_embeddings,
    gflownet_classical=None,
    gflownet_dpo_preference=None,
    all_labels=None,         
    all_annotations=None,   
    target_classes=None,
    cat_map=None
):
    """
    Génère et affiche les UMAP pour les séquences obtenues par les politiques.
    
    En fonction du dataset (COCO ou autre), on utilise soit la fonction collect_embeddings_from_policy
    (pour COCO) soit collect_embeddings_cifar (pour CIFAR, MNIST, etc.).
    
    - Si gflownet_classical est fourni, on trace le graphique pour la politique "classical".
    - Si gflownet_dpo_preference est fourni, on trace le graphique pour la politique "comparison".
    - Si les deux sont fournis, on trace également un graphique combiné.
    """
    num_samples = 500 
    classical_points, classical_labels = None, None
    comparison_points, comparison_labels = None, None

    if config.dataset == "COCO":
        if gflownet_classical is not None:
            classical_points, classical_labels = collect_embeddings_from_policy(
                policy=gflownet_classical,
                embeddings=all_embeddings,
                all_annotations=all_annotations,
                cat_name_to_id_map=cat_map,
                num_samples=num_samples,
                subset_size=config.subset_size,
                device=device
            )
            plot_umap_and_log(
                data_array=classical_points,
                score_array=classical_labels,
                title="UMAP of sequences colored by category combination classical",
                cmap="Reds",
                wandb_key="GFlowNet___/UMAP_GFlowNet_classical"
            )

        if gflownet_dpo_preference is not None:
            comparison_points, comparison_labels = collect_embeddings_from_policy(
                policy=gflownet_dpo_preference,
                embeddings=all_embeddings,
                all_annotations=all_annotations,
                cat_name_to_id_map=cat_map,
                num_samples=num_samples,
                subset_size=config.subset_size,
                device=device
            )
            plot_umap_and_log(
                data_array=comparison_points,
                score_array=comparison_labels,
                title="UMAP of sequences colored by category combination comparison",
                cmap="Reds",
                wandb_key="GFlowNet___/UMAP_gflownet_dpo_preference"
            )

        if (classical_points is not None) and (comparison_points is not None):
            all_points = np.concatenate([classical_points, comparison_points], axis=0)
            all_labels_concat = np.concatenate([classical_labels, comparison_labels], axis=0)
            plot_umap_and_log(
                data_array=all_points,
                score_array=all_labels_concat,
                title="UMAP of sequences colored by category combination",
                cmap="Reds",
                wandb_key="GFlowNet___/UMAP_GFlowNet"
            )

    else:
        # cifar mnist, etc 
        if gflownet_classical is not None:
            classical_points, classical_labels = collect_embeddings_cifar(
                policy=gflownet_classical,
                embeddings=all_embeddings,
                all_labels=all_labels,
                target_classes=target_classes,
                num_samples=num_samples,
                subset_size=config.subset_size,
                device=device
            )
            plot_umap_and_log(
                data_array=classical_points,
                score_array=classical_labels,
                title="UMAP of sequences colored by category combination classical",
                cmap="Reds",
                wandb_key="GFlowNet___/UMAP_GFlowNet_classical"
            )

        if gflownet_dpo_preference is not None:
            comparison_points, comparison_labels = collect_embeddings_cifar(
                policy=gflownet_dpo_preference,
                embeddings=all_embeddings,
                all_labels=all_labels,
                target_classes=target_classes,
                num_samples=num_samples,
                subset_size=config.subset_size,
                device=device
            )
            plot_umap_and_log(
                data_array=comparison_points,
                score_array=comparison_labels,
                title="UMAP of sequences colored by category combination comparison",
                cmap="Reds",
                wandb_key="GFlowNet___/UMAP_gflownet_dpo_preference"
            )

        if (classical_points is not None) and (comparison_points is not None):
            all_points = np.concatenate([classical_points, comparison_points], axis=0)
            all_labels_concat = np.concatenate([classical_labels, comparison_labels], axis=0)
            plot_umap_and_log(
                data_array=all_points,
                score_array=all_labels_concat,
                title="UMAP of sequences colored by category combination",
                cmap="Reds",
                wandb_key="GFlowNet___/UMAP_GFlowNet"
            )