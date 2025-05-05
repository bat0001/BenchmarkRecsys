import math
import torch
import numpy as np
from tqdm import tqdm
import wandb
from collections import Counter
import torch.nn.functional as F
import random
from collections import deque
from utils.sampling import sample_images

def compute_sequence_entropy(seq_list):
    freq = Counter(tuple(seq) for seq in seq_list)
    total = sum(freq.values())
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def compute_image_entropy(seq_list):
    freq_img = Counter()
    total_img = 0
    for seq in seq_list:
        for img_idx in seq:
            freq_img[img_idx] += 1
            total_img += 1
    entropy = 0.0
    for count in freq_img.values():
        p = count / total_img
        entropy -= p * math.log2(p)
    return entropy

def sample_and_log_metrics(trainer, iteration, tag_prefix, train_embeddings, train_labels,
                           target_classes, class_indices, num_samples=256):
    trainer.model.eval()
    seq_list = []
    rewards_list = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            seq_indices = sample_images(
                gflownet=trainer.model,
                embeddings=train_embeddings,
                subset_size=trainer.subset_size,
                top_k=False,
                device=trainer.device
            )
            seq_list.append(seq_indices)

            r = trainer.reward_fn(
                torch.tensor(seq_indices, device=trainer.device).unsqueeze(0),
                train_labels,
                target_classes,
                class_indices
            )
            rewards_list.append(r.item())
    
    rewards_np = np.array(rewards_list)
    reward_min = float(rewards_np.min())
    reward_max = float(rewards_np.max())
    reward_median = float(np.median(rewards_np))
    reward_mean = float(rewards_np.mean())

    wandb.log({
        f"{tag_prefix}/reward_min": reward_min,
        f"{tag_prefix}/reward_max": reward_max,
        f"{tag_prefix}/reward_median": reward_median,
        f"{tag_prefix}/reward_mean": reward_mean,
        f"{tag_prefix}/reward_hist": wandb.Histogram(rewards_np),
    })

    seq_entropy = compute_sequence_entropy(seq_list)
    img_entropy = compute_image_entropy(seq_list)
    distinct_sequences = len(set(tuple(seq) for seq in seq_list))

    wandb.log({
        f"{tag_prefix}/img_entropy": img_entropy,
    })

    trainer.model.train()

    print(f"Reward median: {reward_median}")

    return {
        "Reward Min": reward_min,
        "Reward Max": reward_max,
        "Reward Median": reward_median,
        "Reward Mean": reward_mean,
        "Seq Entropy": seq_entropy,
        "Image Entropy": img_entropy,
        "Distinct Sequences": distinct_sequences
    }

def gflownet_loss_multiclass(selected_indices, rewards, logits, entropy_coeff=0.01, baseline_val=None):
    log_probs = torch.log_softmax(logits, dim=1)
    selected_log_probs = log_probs.gather(1, selected_indices)
    sum_log_probs = selected_log_probs.sum(dim=1)

    advantage = rewards - baseline_val
    loss_pg = - advantage * sum_log_probs

    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)

    loss = loss_pg + entropy_coeff * entropy
    return loss.mean()


def dpo_loss_pairwise(logpA, logpB, preference_mask, beta=1.0):
    delta = beta * (logpA - logpB)
    #verifier 
    loss_pref_1 = -torch.log(torch.sigmoid(delta) + 1e-10)
    loss_pref_0 = -torch.log(torch.sigmoid(-delta) + 1e-10)
    loss = preference_mask.float() * loss_pref_1 + (1 - preference_mask.float()) * loss_pref_0
    return loss.mean()


def logprob_of_subset(logits, subset_indices):
    log_probs = F.log_softmax(logits, dim=1)  # [batch_size, num_classes]
    chosen_log_probs = torch.gather(log_probs, 1, subset_indices)
    return chosen_log_probs.sum(dim=1)

class ReplayBuffer:
    def __init__(self, maxlen=1000):
        self.top_buffer = deque(maxlen=maxlen)
        self.bottom_buffer = deque(maxlen=maxlen)

    def add_sequence(self, subset_indices, reward, top_threshold, bottom_threshold):
        if reward > top_threshold:
            self.top_buffer.append(subset_indices)
        elif reward < bottom_threshold:
            self.bottom_buffer.append(subset_indices)

    def sample_pair(self, batch_size):
        top_size = min(len(self.top_buffer), batch_size)
        bottom_size = min(len(self.bottom_buffer), batch_size)

        if top_size < batch_size or bottom_size < batch_size:
            return None, None

        selected_top = random.sample(self.top_buffer, batch_size)
        selected_bottom = random.sample(self.bottom_buffer, batch_size)

        A = torch.stack(selected_top, dim=0)  # [batch_size, subset_size]
        B = torch.stack(selected_bottom, dim=0)
        return A, B
    

class GFlowNetTrainer:
    def __init__(self, model, optimizer, device, reward_fn, class_indices,
                 subset_size, entropy_coeff=0.01, initial_temp=2.0, final_temp=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.reward_fn = reward_fn
        self.class_indices = class_indices
        self.subset_size = subset_size
        self.entropy_coeff = entropy_coeff
        self.alpha = 0.99
        self.moving_average_loss = 0.0

        self.initial_temp = initial_temp
        self.final_temp = final_temp

        self.distinct_relevant_sequences = set()

        self.baseline_val = 0.0
        self.baseline_val_comparison = 0.0
        self.baseline_alpha = 0.9

        self.iterations_classical = []
        self.seq_relevant_classical = []
        self.images_seen_classical = []
        self.reward_vs_images_classical = []

        self.iterations_comparison = []
        self.seq_relevant_comparison = []
        self.images_seen_comparison = []
        self.reward_vs_images_comparison = []

        self.cumulative_images_classical = 0
        self.cumulative_images_comparison = 0

        self.replay_buffer = ReplayBuffer(maxlen=2000)

    def sample_and_log_metrics(self, iteration, tag_prefix, train_embeddings, train_labels,
                               target_classes, class_indices, num_samples=256):
        metrics_dict = sample_and_log_metrics(self, iteration, tag_prefix,
                                              train_embeddings, train_labels,
                                              target_classes, class_indices, num_samples)
        return metrics_dict

    def train_iterations_classical(self, train_embeddings, train_labels, target_classes,
                                   num_iterations, batch_size_train, logger):
        self.model.train()

        for it in tqdm(range(num_iterations), desc="Training Multilabeled Classical GFlowNet"):
            temp = (self.initial_temp + (self.final_temp - self.initial_temp) *
                    (it / (num_iterations - 1))) if num_iterations > 1 else self.initial_temp

            self.cumulative_images_classical += batch_size_train

            batch_indices = torch.randint(0, train_embeddings.size(0), (batch_size_train,))
            batch_embeddings = train_embeddings[batch_indices].to(self.device)

            logits = self.model(batch_embeddings)
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=1) + 1e-8
            selected_indices = torch.multinomial(probs, self.subset_size, replacement=False)

            rewards = self.reward_fn(selected_indices, train_labels, target_classes, self.class_indices).to(self.device)
            current_mean_reward = rewards.mean().item()

            self.baseline_val = (self.baseline_alpha * self.baseline_val
                                 + (1 - self.baseline_alpha) * current_mean_reward)

            loss = gflownet_loss_multiclass(selected_indices, rewards, logits,
                                             entropy_coeff=self.entropy_coeff,
                                             baseline_val=self.baseline_val)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.moving_average_loss = (self.alpha * self.moving_average_loss
                                        + (1 - self.alpha) * loss.item())

            self.iterations_classical.append(it)
            threshold_relev = 1.0
            relevant_count = (rewards > threshold_relev).sum().item()
            self.seq_relevant_classical.append(relevant_count) 
            self.images_seen_classical.append(self.cumulative_images_classical)
            self.reward_vs_images_classical.append(current_mean_reward)

            if it % 1000 == 0 and it > 0:
                num_correct = (rewards > 0).sum().item()
                print(f"Iteration {it}, Temp: {temp:.2f}, Loss: {loss.item():.4f}, "
                      f"Mean Moving Loss: {self.moving_average_loss:.4f}, "
                      f"Mean Reward: {rewards.mean().item():.4f}, Correct Sets: {num_correct}/{batch_size_train}")
                logger.log({
                    "GFlowNet_Classical/loss": loss.item(),
                    "GFlowNet_Classical/moving_avg_loss": self.moving_average_loss,
                    "GFlowNet_Classical/mean_reward": rewards.mean().item(),
                })

            if it % 1000 == 0 and it > 0:
                _ = self.sample_and_log_metrics(
                    iteration=it,
                    tag_prefix="GFlowNet_Classical/Online",
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    target_classes=target_classes,
                    class_indices=self.class_indices,
                    num_samples=256
                )

    def train_iterations_binary_preference(self, train_embeddings, train_labels, target_classes,
                                   num_iterations, batch_size_train, logger):
        self.model.train()

        for it in tqdm(range(num_iterations), desc="Training Multilabeled Classical GFlowNet"):
            temp = (self.initial_temp + (self.final_temp - self.initial_temp) *
                    (it / (num_iterations - 1))) if num_iterations > 1 else self.initial_temp

            self.cumulative_images_classical += batch_size_train

            batch_indices = torch.randint(0, train_embeddings.size(0), (batch_size_train,))
            batch_embeddings = train_embeddings[batch_indices].to(self.device)

            logits = self.model(batch_embeddings)
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=1) + 1e-8
            selected_indices = torch.multinomial(probs, self.subset_size, replacement=False)

            rewards = self.reward_fn(selected_indices, train_labels, target_classes, self.class_indices).to(self.device)
            current_mean_reward = rewards.mean().item()

            self.baseline_val = (self.baseline_alpha * self.baseline_val
                                 + (1 - self.baseline_alpha) * current_mean_reward)

            loss = gflownet_loss_multiclass(selected_indices, rewards, logits,
                                             entropy_coeff=self.entropy_coeff,
                                             baseline_val=self.baseline_val)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.moving_average_loss = (self.alpha * self.moving_average_loss
                                        + (1 - self.alpha) * loss.item())

            self.iterations_classical.append(it)
            threshold_relev = 1.0
            relevant_count = (rewards > threshold_relev).sum().item()
            self.seq_relevant_classical.append(relevant_count) 
            self.images_seen_classical.append(self.cumulative_images_classical)
            self.reward_vs_images_classical.append(current_mean_reward)

            if it % 1000 == 0 and it > 0:
                num_correct = (rewards > 0).sum().item()
                print(f"Iteration {it}, Temp: {temp:.2f}, Loss: {loss.item():.4f}, "
                      f"Mean Moving Loss: {self.moving_average_loss:.4f}, "
                      f"Mean Reward: {rewards.mean().item():.4f}, Correct Sets: {num_correct}/{batch_size_train}")
                logger.log({
                    "GFlowNet_Classical/loss": loss.item(),
                    "GFlowNet_Classical/moving_avg_loss": self.moving_average_loss,
                    "GFlowNet_Classical/mean_reward": rewards.mean().item(),
                })

            if it % 1000 == 0 and it > 0:
                _ = self.sample_and_log_metrics(
                    iteration=it,
                    tag_prefix="GFlowNet_Classical/Online",
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    target_classes=target_classes,
                    class_indices=self.class_indices,
                    num_samples=256
                )

    def train_iterations_comparison(self, train_embeddings, train_labels, target_classes,
                                    num_iterations, batch_size_train, logger):
        self.model.train()

        beta = 50.0
        fixed_temp = 0.3
        N_pool = 30
        local_entropy_coeff = 0.0

        for it in tqdm(range(num_iterations), desc="Training Comparison (Active Mix)"):
            self.cumulative_images_comparison += batch_size_train

            batch_indices = torch.randint(0, train_embeddings.size(0), (batch_size_train,))
            batch_embeddings = train_embeddings[batch_indices].to(self.device)

            with torch.no_grad():
                logits_batch = self.model(batch_embeddings)  # [batch_size, num_classes]
                scaled_logits_batch = logits_batch / fixed_temp
                probs_batch = torch.softmax(scaled_logits_batch, dim=1) + 1e-12

            pool_of_sequences = []
            rewards_array = []
            for _ in range(N_pool):
                seq_indices = torch.multinomial(probs_batch, self.subset_size, replacement=False)
                pool_of_sequences.append(seq_indices)
                r = self.reward_fn(seq_indices, train_labels, target_classes, self.class_indices)
                rewards_array.append(r.unsqueeze(0))

            R_pool = torch.cat(rewards_array, dim=0).t().to(self.device)

            best_indices = []
            worst_indices = []
            mid_indices = []

            for i in range(batch_size_train):
                row = R_pool[i]
                sorted_vals, sorted_ids = torch.sort(row)
                bot_id = sorted_ids[0]
                top_id = sorted_ids[-1]
                mid_id = sorted_ids[len(sorted_ids)//2]

                best_seq = pool_of_sequences[top_id][i]
                worst_seq = pool_of_sequences[bot_id][i]
                mid_seq  = pool_of_sequences[mid_id][i]

                best_indices.append(best_seq)
                worst_indices.append(worst_seq)
                mid_indices.append(mid_seq)

            selected_indices_top = torch.stack(best_indices, dim=0).to(self.device)
            selected_indices_bot = torch.stack(worst_indices, dim=0).to(self.device)
            selected_indices_mid = torch.stack(mid_indices,   dim=0).to(self.device)

            A_concat = torch.cat([selected_indices_top,  selected_indices_top ], dim=0)
            B_concat = torch.cat([selected_indices_bot,  selected_indices_mid ], dim=0)

            with torch.no_grad():
                rewards_A = self.reward_fn(A_concat, train_labels, target_classes, self.class_indices).to(self.device)
                rewards_B = self.reward_fn(B_concat, train_labels, target_classes, self.class_indices).to(self.device)

            preference_mask = (rewards_A > rewards_B)
            ratio_winners = preference_mask.sum().item() / (2.0 * batch_size_train)

            self.optimizer.zero_grad()
            logits_for_grad = self.model(batch_embeddings)
            
            half_size = batch_size_train
            A1 = A_concat[:half_size]
            B1 = B_concat[:half_size]
            pref_mask1 = preference_mask[:half_size]

            logpA1 = logprob_of_subset(logits_for_grad, A1)
            logpB1 = logprob_of_subset(logits_for_grad, B1)

            loss_dpo_1 = dpo_loss_pairwise(logpA1, logpB1, pref_mask1, beta=beta)

            A2 = A_concat[half_size:]
            B2 = B_concat[half_size:]
            pref_mask2 = preference_mask[half_size:]

            logpA2 = logprob_of_subset(logits_for_grad, A2)
            logpB2 = logprob_of_subset(logits_for_grad, B2)

            loss_dpo_2 = dpo_loss_pairwise(logpA2, logpB2, pref_mask2, beta=beta)

            loss_dpo = loss_dpo_1 + loss_dpo_2

            log_probs = F.log_softmax(logits_for_grad, dim=1)
            probs = torch.softmax(logits_for_grad, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()

            loss_total = loss_dpo - local_entropy_coeff * entropy
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            current_loss = loss_dpo.item()
            self.moving_average_loss = (
                self.alpha * self.moving_average_loss
                + (1 - self.alpha) * current_loss
            )

            mean_reward_A = rewards_A.mean().item()
            mean_reward_B = rewards_B.mean().item()
            mean_reward = 0.5 * (mean_reward_A + mean_reward_B)

            if it % 2000 == 0 and it > 0:
                n_print = 3
                for _ in range(n_print):
                    i = random.randint(0, batch_size_train-1)
                    top_seq = selected_indices_top[i].tolist()
                    mid_seq = selected_indices_mid[i].tolist()
                    bot_seq = selected_indices_bot[i].tolist()

                    top_seq_tensor = torch.tensor([top_seq], dtype=torch.long, device=self.device)
                    mid_seq_tensor = torch.tensor([mid_seq], dtype=torch.long, device=self.device)
                    bot_seq_tensor = torch.tensor([bot_seq], dtype=torch.long, device=self.device)

                    r_top = self.reward_fn(top_seq_tensor, train_labels, target_classes, self.class_indices)
                    r_mid = self.reward_fn(mid_seq_tensor, train_labels, target_classes, self.class_indices)
                    r_bot = self.reward_fn(bot_seq_tensor, train_labels, target_classes, self.class_indices)

                    print(f"Example {i}: TOP seq={top_seq}, reward={r_top.item():.1f} "
                        f"| MID seq={mid_seq}, reward={r_mid.item():.1f} "
                        f"| BOT seq={bot_seq}, reward={r_bot.item():.1f}")
                
            if it % 1000 == 0:
                print(f"[ActiveMixDPO] Iter {it}, Loss: {loss_dpo.item():.4f}, MoveLoss: {self.moving_average_loss:.4f}, "
                      f"MeanReward: {mean_reward:.2f}, Beta={beta:.1f}, ratioWin={ratio_winners:.2f}")
                logger.log({
                    "Comparison/loss": loss_dpo.item(),
                    "Comparison/moving_avg_loss": self.moving_average_loss,
                    "Comparison/mean_reward": mean_reward,
                    "Comparison/beta": beta,
                    "Comparison/ratio_winners": ratio_winners
                })

            if it % 1000 == 0 and it > 0:
                _ = self.sample_and_log_metrics(
                    iteration=it,
                    tag_prefix="GFlowNet_Comparison/Online",
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    target_classes=target_classes,
                    class_indices=self.class_indices,
                    num_samples=256
                )