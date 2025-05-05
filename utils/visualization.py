import umap
import wandb
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from collections import Counter
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

def visualize_recommendations(indices, dataset, num_images=4):
    plt.figure(figsize=(20, 5))
    for i, idx in enumerate(indices[:num_images]):
        plt.subplot(1, num_images, i + 1)
        img = dataset.data[idx]
        if img.shape[0] == 3 and img.shape[1] == 32 and img.shape[2] == 32:
            img = np.transpose(img, (1,2,0))
        elif img.shape[0] == 32 and img.shape[1] == 3 and img.shape[2] == 32:
            img = np.transpose(img, (0,2,1))
        elif img.shape[0] == 32 and img.shape[1] == 32 and img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        plt.imshow(img)
        plt.title(f"Image {idx} - {dataset.classes[dataset.targets[idx]]}")
        plt.axis('off')
    plt.suptitle(f"Multiclass Recommendations ({num_images} Images)")
    plt.show()

def log_recommendations_to_wandb(indices, dataset, key):
    images = [
        wandb.Image(dataset.data[idx], caption=f"Image {idx} - {dataset.classes[dataset.targets[idx]]}")
        for idx in indices
    ]
    wandb.log({key: images})

def compute_image_selection_probabilities(model, all_embeddings, device):
    """
    Calcule, pour chaque image de l'ensemble filtré, la probabilité moyenne
    d'être sélectionnée. Pour cela, on passe l'ensemble des embeddings dans le
    modèle GFlowNet qui renvoie des logits de shape [N, pool_size] puis on applique un softmax
    ligne par ligne et on fait la moyenne sur toutes les lignes (i.e. sur toutes les requêtes).
    
    Args:
        model (nn.Module): Le modèle (GFlowNet) entraîné.
        all_embeddings (Tensor): Les embeddings de shape [N, latent_dim] (N = nombre d'images).
        device (torch.device): Le device à utiliser (CPU ou GPU).
    
    Returns:
        np.array: Un tableau numpy de shape [pool_size] contenant la probabilité moyenne 
                  pour chaque image (dans l’ensemble de candidats).
    """
    model.eval()
    with torch.no_grad():
        logits = model(all_embeddings.to(device))
        probs = torch.softmax(logits, dim=1)
    image_probs = probs.mean(dim=0)
    return image_probs.cpu().numpy()

def plot_image_probability_by_index(image_probs, scale_factor=1, title="Selection Probability per Image (Scaled)", figsize=(12,6)):
    """

    Args:
        image_probs (np.array): Tableau de shape [N] contenant la probabilité de sélection
                                pour chaque image.
        scale_factor (float): Facteur de multiplication pour "agrandir" les valeurs.
        title (str): Titre du graphique.
        figsize (tuple): Taille de la figure.
    
    Returns:
        matplotlib.figure.Figure: La figure contenant le scatter plot.
    """
    scaled_probs = image_probs
    fig, ax = plt.subplots(figsize=figsize)
    indices = np.arange(len(scaled_probs))
    ax.scatter(indices, scaled_probs, s=10, alpha=0.6, color='blue')
    ax.set_xlabel("Image Index")
    ax.set_ylabel(f"Selection Probability ")
    ax.set_title(title)
    ax.set_ylim(0, 0.1)
    plt.tight_layout()
    return fig

def print_coco_class_distribution_subset(train_dataset):
    coco = train_dataset.coco
    img_ids_subset = train_dataset.ids
    ann_ids_subset = coco.getAnnIds(imgIds=img_ids_subset)
    all_annots = coco.loadAnns(ann_ids_subset)

    instance_counts = Counter()
    for annot in all_annots:
        cat_id = annot['category_id']
        instance_counts[cat_id] += 1
    
    print("Distribution des classes (Nombre d'instances) dans le sous-ensemble filtré :")
    for cat_id in sorted(instance_counts.keys()):
        class_name = coco.loadCats(cat_id)[0]['name']
        count = instance_counts[cat_id]
        print(f"  {class_name}: {count} instances")

def print_class_distribution(dataset, dataset_name):
    if dataset_name in ["CIFAR-10", "CIFAR-100", "MNIST", "FashionMNIST"]:
        labels = dataset.targets
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        label_counts = Counter(labels)
        
        class_names = getattr(dataset, 'classes', None)
        if class_names is None:
            print("L'attribut 'classes' n'est pas défini pour ce dataset.")
            return
        
        print(f"Distribution des classes pour {dataset_name}:")
        for class_idx, count in label_counts.items():
            class_name = class_names[class_idx] if class_names else f"Classe {class_idx}"
            print(f"  {class_name}: {count} images")
    else:
        print(f"Le dataset {dataset_name} n'est pas pris en charge par cette fonction.")

def plot_image_probability_by_objective2(image_probs, all_annotations, objectives, cat_map, 
                                          scale_factor=1e5, 
                                          title="Selection Probability per Image with Objectives", 
                                          figsize=(12,6)):
   
    # color_map = {
    #     "person": "yellow",
    #     "skateboard": "orange",
    #     "backpack": "red"
    # }
    color_map = {
        "car": "yellow",
        "traffic lights": "orange",
        "bus": "red"
    }
    default_color = "blue"
    
    N = len(image_probs)
    indices = np.arange(N)
    
    scaled_probs = image_probs 
    
    point_colors = []
    for i in range(N):
        ann_list = all_annotations[i] 
        assigned_color = default_color
        max_reward = 0.0
        for obj, reward_val in objectives.items():
            cat_id = cat_map.get(obj)
            if cat_id is None:
                continue
            if any(ann.get("category_id") == cat_id for ann in ann_list):
                if reward_val > max_reward:
                    max_reward = reward_val
                    assigned_color = color_map.get(obj, default_color)
        point_colors.append(assigned_color)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(indices, scaled_probs, s=10, alpha=0.6, c=point_colors)
    ax.set_xlabel("Index de l'image")
    ax.set_ylabel(f"Probabilité de sélection")
    ax.set_title(title)

    ax.set_yscale("log")
    # ax.set_ylim(0, 0.09)
    
    plt.tight_layout()
    return fig

def generate_color_map(objectives, colormap_name='rainbow'):
    
    rewards = np.array(list(objectives.values()))
    if rewards.max() == rewards.min():
        normalized = np.ones_like(rewards) * 0.5
    else:
        normalized = (rewards - rewards.min()) / (rewards.max() - rewards.min())
    cmap = cm.get_cmap(colormap_name)
    color_mapping = {}
    for key, norm_value in zip(objectives.keys(), normalized):
        color_mapping[key] = mcolors.to_hex(cmap(norm_value))
    return color_mapping

def plot_image_probability_by_objective(image_probs, all_annotations, objectives, cat_map, 
                                        scale_factor=1e5, 
                                        title="Selection Probability per Image with Objectives", 
                                        figsize=(12,6),
                                        colormap_name='rainbow',  
                                        invert=False ):
    """
    Trace un scatter plot montrant la probabilité de sélection des images, 
    en assignant à chaque point une couleur dépendante de l'objectif présent
    dans l'image et de la reward associée.

    Paramètres:
    -----------
    image_probs : array-like
        Probabilités de sélection pour chaque image.
    all_annotations : list
        Liste d'annotations pour chaque image.
    objectives : dict
        Dictionnaire d'objectifs et leurs rewards (ex: {"person":1.0, "skateboard":2.0, ...}).
    cat_map : dict
        Mapping de nom de catégorie vers identifiant (ex: {"person":1, "skateboard":24, ...}).
    scale_factor : float, optionnel
        Facteur d'échelle si nécessaire (non utilisé ici).
    title : str
        Titre du graphique.
    figsize : tuple
        Taille de la figure.
    colormap_name : str
        Nom du colormap à utiliser pour générer automatiquement les couleurs.
        
    Return:
    fig : matplotlib.figure.Figure
    """
    generated_color_map = generate_color_map(objectives, colormap_name=colormap_name)
    default_color = "blue"  
    
    N = len(image_probs)
    indices = np.arange(N)
    
    scaled_probs = image_probs  #* scale_factor 
    
    point_colors = []
    for i in range(N):
        ann_list = all_annotations[i]
        assigned_color = default_color
        max_reward_found = 0.0
        for obj, reward_val in objectives.items():
            cat_id = cat_map.get(obj)
            if cat_id is None:
                continue
            if any(ann.get("category_id") == cat_id for ann in ann_list):
                if reward_val > max_reward_found:
                    max_reward_found = reward_val
                    assigned_color = generated_color_map.get(obj, default_color)
        point_colors.append(assigned_color)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(indices, scaled_probs, s=10, alpha=0.6, c=point_colors)
    ax.set_xlabel("Index de l'image")
    ax.set_ylabel("Probabilité de sélection")
    ax.set_title(title)
    ax.set_yscale("log")
    
    plt.tight_layout()
    return fig

def plot_image_probability_by_targets_cifar(image_probs, all_annotations, objectives, class_names, 
                                              scale_factor=1e5, 
                                              title="Selection Probability per Image with Objectives", 
                                              figsize=(12,6)):
    """
    Pour CIFAR (ou datasets similaires) :
      - image_probs : tableau (numpy array ou liste) de probabilités de sélection pour chaque image.
      - all_annotations : liste (ou tableau) d'indices de classes (entiers) pour chaque image.
      - objectives : dictionnaire définissant les objectifs avec par exemple {"truck": 10.0, "airplane": 20.0}
      - class_names : liste des noms de classes dans l'ordre (ex: dataset.classes)
      
    Les points correspondant aux images dont la classe figure dans les objectifs seront colorés selon une color_map.
    Les autres seront affichés en couleur par défaut.
    """
    
    color_map = {
        "truck": "yellow",
        "airplane": "orange",
    }
    default_color = "blue"
    
    N = len(image_probs)
    indices = np.arange(N)
    
    scaled_probs = np.array(image_probs) 
    
    point_colors = []
    for i in range(N):
        target_label = all_annotations[i]
        class_name = class_names[target_label]
        if class_name in objectives:
            assigned_color = color_map.get(class_name, default_color)
        else:
            assigned_color = default_color
        point_colors.append(assigned_color)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(indices, scaled_probs, s=10, alpha=0.6, c=point_colors)
    ax.set_xlabel("Index de l'image")
    ax.set_ylabel(f"Probabilité de sélection")
    ax.set_title(title)
    
    ax.set_yscale("log")
    # ax.set_ylim(1e-6, 0.09)
    
    plt.tight_layout()
    return fig


def plot_umap_and_log(
    data_array,          
    score_array,         
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    cmap="coolwarm",     
    title="UMAP Plot",
    wandb_key="UMAP_ScorePlot"
):
    """
    1) Applique UMAP pour réduire data_array en 2D.
    2) Trace un scatter plot en colorant les points selon score_array, 
       avec un colormap qui rend visible même les faibles scores.
    3) Loggue la figure dans wandb.
    """
    if isinstance(data_array, torch.Tensor):
        data_array = data_array.cpu().numpy()
    if isinstance(score_array, torch.Tensor):
        score_array = score_array.cpu().numpy()

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding_2d = reducer.fit_transform(data_array)  # shape [N, 2]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=score_array, cmap=cmap, alpha=0.7)
    plt.colorbar(sc, label="Score")
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    wandb.log({wandb_key: wandb.Image(plt)})
    plt.close()

def smooth_data_savgol(x, y, window_length=51, polyorder=3):
    """
    Lisse les données x et y avec un filtre de Savitzky–Golay.

    Paramètres:
      - x : array ou liste (axe des x)
      - y : array ou liste (valeurs à lisser)
      - window_length : taille de la fenêtre (doit être impair)
      - polyorder : degré du polynôme utilisé pour le filtrage

    Retourne:
      - x, y_lisse : x inchangé et y lissé
    """
    x = np.array(x)
    y = np.array(y)
    if len(y) < window_length:
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
    if window_length < 3:
        return x, y  
    y_smooth = savgol_filter(y, window_length, polyorder)
    return x, y_smooth

def plot_comparison(
    x_vals_classical, y_vals_classical, 
    x_vals_comparison, y_vals_comparison,
    title, x_label, y_label, wandb_key,
    x_vals_random=None, y_vals_random=None,      
    max_reward=None                              
):
    """
    Trace un graphe de comparaison entre :
      - Méthode "Classical" (marker='o')
      - Méthode "Comparison" (marker='x')
      - Eventuellement, baseline aléatoire (marker='^', vert)
      - Eventuellement, ligne horizontale (max_reward) en rouge pointillée
    """
    
    plt.figure(figsize=(8, 5))
    
    x_classical, y_classical = smooth_data_savgol(x_vals_classical, y_vals_classical, window_length=51, polyorder=3)
    x_comparison, y_comparison = smooth_data_savgol(x_vals_comparison, y_vals_comparison, window_length=51, polyorder=3)
    
    plt.plot(x_classical, y_classical, 
             label="Classical", marker='o', markersize=2, color='blue', linewidth=0.5)
    plt.plot(x_comparison, y_comparison, 
             label="Comparison", marker='x', markersize=2, color='orange', linewidth=0.5)
    
    if (x_vals_random is not None) and (y_vals_random is not None):
        x_random, y_random = smooth_data_savgol(x_vals_random, y_vals_random, window_length=51, polyorder=3)
        plt.plot(x_random, y_random,
                 label="Random Baseline", marker='^', markersize=2, color='green', linewidth=0.5)
    
    if max_reward is not None:
        plt.axhline(y=max_reward, color='red', linestyle='--',
                    label=f"Max. Mean Reward = {max_reward}")
    
    plt.xscale('log')
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    wandb.log({wandb_key: wandb.Image(plt)})
    plt.close()


def plot_comparison_iterations(
    x_vals_classical, y_vals_classical,
    x_vals_comparison, y_vals_comparison,
    x_vals_random=None, y_vals_random=None,
    max_reward=None,
    title="Comparison Plot (Iterations)",
    x_label="Iterations",
    y_label="Reward",
    wandb_key="ComparisonPlot/RewardVsIterations"
):
    """
    Trace la comparaison entre deux méthodes (Classical et Comparison)
    en fonction des itérations (x-axis) et de la reward (y-axis).
    """
    
    plt.figure(figsize=(8, 5))
    
    x_classical, y_classical = smooth_data_savgol(x_vals_classical, y_vals_classical, window_length=51, polyorder=3)
    x_comparison, y_comparison = smooth_data_savgol(x_vals_comparison, y_vals_comparison, window_length=51, polyorder=3)
    
    plt.plot(x_classical, y_classical, label="Classical", marker='o', markersize=2, color='blue', linewidth=0.5)
    plt.plot(x_comparison, y_comparison, label="Comparison", marker='x', markersize=2, color='orange', linewidth=0.5)
    
    if (x_vals_random is not None) and (y_vals_random is not None):
        x_random, y_random = smooth_data_savgol(x_vals_random, y_vals_random, window_length=51, polyorder=3)
        plt.plot(x_random, y_random, label="Random Baseline", marker='^', markersize=2, color='green', linewidth=0.5)
    
    if max_reward is not None:
        plt.axhline(y=max_reward, color='red', linestyle='--',
                    label=f"Max. Mean Reward = {max_reward}")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    wandb.log({wandb_key: wandb.Image(plt)})
    plt.close()