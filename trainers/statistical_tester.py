from tqdm import tqdm
from scipy import stats
from utils.metrics import evaluate_model

class StatisticalTester:
    def __init__(self, model_classical, model_comparison, embeddings, labels, target_classes, class_indices, subset_size, device):
        self.model_classical = model_classical
        self.model_comparison = model_comparison
        self.embeddings = embeddings
        self.labels = labels
        self.target_classes = target_classes
        self.class_indices = class_indices
        self.subset_size = subset_size
        self.device = device
    
    def perform_tests(self, num_samples=100, logger=None):
        classical_metrics = {'Precision': [], 'Recall': [], 'F1-Score': [], 'Exact Match Ratio': []}
        comparison_metrics = {'Precision': [], 'Recall': [], 'F1-Score': [], 'Exact Match Ratio': []}
        
        for _ in tqdm(range(num_samples), desc="Collection metrics for tests"):
            metrics = evaluate_model(
                self.model_classical, 
                self.embeddings, 
                self.labels, 
                self.target_classes, 
                self.class_indices, 
                self.subset_size, 
                self.device, 
                top_k=True
            )
            for key in classical_metrics:
                classical_metrics[key].append(metrics[key])
            
            metrics = evaluate_model(
                self.model_comparison, 
                self.embeddings, 
                self.labels, 
                self.target_classes, 
                self.class_indices, 
                self.subset_size, 
                self.device, 
                top_k=True
            )
            for key in comparison_metrics:
                comparison_metrics[key].append(metrics[key])
        
        for metric in classical_metrics.keys():
            t_stat, p_val = stats.ttest_ind(classical_metrics[metric], comparison_metrics[metric], equal_var=False)
            print(f"Test t pour {metric} : t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
            if logger:
                logger.log({
                    f"Statistical_Test/{metric}_t_statistic": t_stat,
                    f"Statistical_Test/{metric}_p_value": p_val
                })