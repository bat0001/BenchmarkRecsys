import torch

def sample_images(gflownet, embeddings, subset_size=4, top_k=True, device='cpu'):
   
    gflownet.eval()
    with torch.no_grad():
        logits = gflownet(embeddings.to(device))
        if top_k:
            _, selected_indices = torch.topk(logits, subset_size, dim=1)
        else:
            probs = torch.softmax(logits, dim=1) + 1e-8
            selected_indices = torch.multinomial(probs, subset_size, replacement=False)
        rec_indices = selected_indices[0].cpu().numpy()
    return rec_indices

def sample_many_sequences(gflownet, embeddings, num_samples=4, subset_size=3, device='cpu', top_k=False):
  
    gflownet.eval()
    seqs = []
    for _ in range(num_samples):
        seq = sample_images(gflownet, embeddings, subset_size=subset_size, top_k=top_k, device=device)
        seqs.append(seq) 
    return seqs


def sample_one_action(logits, mask=None, temp=1.0, device='cpu'):
    """
    Sélectionne 1 action (indice d'image) à partir de 'logits'.
    
    - logits : tenseur (1, pool_size) OU (pool_size,)
               représentant les logits non normalisés pour chaque action/image
    - mask   : tenseur bool (pool_size,) indiquant quelles actions sont valides
               (True = action interdite/invalide), ou None si pas de masque
    - temp   : factor de température pour le softmax
    - device : 'cpu' ou 'cuda'
    
    Retourne : action (int)
    """
    if len(logits.shape) == 2:
        logits = logits[0]  

    if mask is not None:
        cloned_logits = logits.clone()
        cloned_logits[mask] = -1e20
        logits = cloned_logits

    scaled_logits = logits / temp
    probs = torch.softmax(scaled_logits, dim=-1)  # shape (pool_size,)

    action = torch.multinomial(probs, num_samples=1).item()
    return action