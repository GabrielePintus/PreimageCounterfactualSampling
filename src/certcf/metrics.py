import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

class TabularCounterfactualEvaluator:
    """
    Evaluator for Tabular Counterfactual Explanations.
    Computes:
    - Validity (Does it flip the classifier?)
    - Distances (L0, L1, L2 norms combining continuous and categorical logic)
    - Plausibility (Manifold adherence using k-NN and Local Outlier Factor)
    """
    def __init__(self, model, X_train_tensor, num_cols, cat_cols, device='cpu'):
        self.model = model.eval().to(device)
        self.device = device
        
        # Determine the physical column indices from boolean or string mapping if necessary
        # Usually we just expect list of integer indices for numpy slicing.
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        # Precompute embeddings for all training data for Plausibility Metrics
        with torch.no_grad():
            self.Z_train = X_train_tensor.cpu().numpy()
            
        # Fit k-NN for plausibility
        self.knn_5 = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.knn_5.fit(self.Z_train)
        
        # Fit LOF (Local Outlier Factor) for plausibility
        # novelty=True allows us to predict the LOF score on new/unseen counterfactuals
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        self.lof.fit(self.Z_train)
        
        # Precompute standard deviations for continuous features to normalize L1/L2 distances
        X_numpy = X_train_tensor.cpu().numpy()
        self.num_std = np.std(X_numpy[:, num_cols], axis=0)
        # Avoid division by zero for constant features
        self.num_std[self.num_std == 0] = 1e-8 

    def evaluate(self, x_orig, x_cf, target_class):
        """
        Evaluate a single generated counterfactual.
        
        Args:
            x_orig: 1D tensor representing the original factual tabular instance
            x_cf: 1D tensor representing the generated counterfactual tabular instance
            target_class: The desired predicted class (int)
            
        Returns:
            dict containing all metric scores
        """
        x_orig_np = x_orig.cpu().numpy() if isinstance(x_orig, torch.Tensor) else x_orig
        x_cf_np = x_cf.cpu().numpy() if isinstance(x_cf, torch.Tensor) else x_cf
        
        metrics = {}
        
        # --- 1. VALIDITY ---
        with torch.no_grad():
            x_cf_tensor = torch.tensor(x_cf_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x_cf_tensor)
            pred = logits.argmax(dim=1).item()
        
        metrics['validity'] = int(pred == target_class)
        metrics['predicted_class'] = pred
        
        # --- 2. DISTANCES (L0, L1, L2) ---
        # L0: Sparsity (Total number of features changed)
        # We round to avoid floating point precision issues on continuous variables
        is_diff = ~np.isclose(x_orig_np, x_cf_np, atol=1e-5)
        metrics['L0_distance'] = int(np.sum(is_diff))
        
        # Feature differences
        diff_num = np.abs(x_orig_np[self.num_cols] - x_cf_np[self.num_cols])
        # Categorical differences (Hamming / Mismatch indicator)
        diff_cat = x_orig_np[self.cat_cols] != x_cf_np[self.cat_cols]
        
        # L1: Sum of absolute differences
        # Normalize continuous difference by training standard deviation (similar to Gower)
        l1_num = np.sum(diff_num / self.num_std)
        l1_cat = np.sum(diff_cat)
        metrics['L1_distance'] = float(l1_num + l1_cat)
        
        # L2: Sqrt of sum of squared differences
        l2_num = np.sum((diff_num / self.num_std)**2)
        l2_cat = np.sum(diff_cat) # squared mismatch is still 0/1
        metrics['L2_distance'] = float(np.sqrt(l2_num + l2_cat))
        
        # --- 3. PLAUSIBILITY (Manifold Adherence) ---
        # We measure plausibility in the embedding space to capture complex hierarchical feature correlations
        with torch.no_grad():
            z_cf = x_cf_tensor.cpu().numpy()
        
        # 3a. k-NN Distance (Density/Proximity to Training Manifold)
        # Lower is better (closer to the training manifold)
        dists, _ = self.knn_5.kneighbors(z_cf)
        metrics['plausibility_knn_5_dist'] = float(dists.mean())
        
        # 3b. Local Outlier Factor (LOF) Score
        # score_samples returns opposite of anomaly score (higher / closer to 0 is better, highly negative is an outlier)
        metrics['plausibility_lof_score'] = float(self.lof.score_samples(z_cf)[0])
        
        return metrics
