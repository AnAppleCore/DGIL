import numpy as np
import torch
import torch.distributions as dist
from sklearn.cluster import KMeans


class CovarianceDist:
    def __init__(self, feature_dim, device):
        self.mean = torch.zeros(feature_dim).to(device)
        self.cov = torch.zeros((feature_dim, feature_dim)).to(device)
        self.num_samples = 0
        self.device = device

    def init_from(self, mean, cov, num_samples):
        """Set the initial mean and covariance."""
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            raise ValueError("Initial mean contains NaN or inf values.")
        if torch.isnan(cov).any() or torch.isinf(cov).any():
            raise ValueError("Initial covariance contains NaN or inf values.")
        self.mean = mean
        self.cov = cov
        self.num_samples = num_samples

    def update(self, new_mean, new_cov, new_samples):
        """Update the mean, covariance, and sample count using the new data."""
        if torch.isnan(new_mean).any() or torch.isinf(new_mean).any():
            raise ValueError("New mean contains NaN or inf values.")
        if torch.isnan(new_cov).any() or torch.isinf(new_cov).any():
            raise ValueError("New covariance contains NaN or inf values.")
        total_samples = self.num_samples + new_samples
        self.mean = (self.num_samples * self.mean + new_samples * new_mean) / total_samples
        diff = new_mean - self.mean
        self.cov = (self.num_samples * self.cov + new_samples * new_cov) / total_samples
        self.cov += (self.num_samples * new_samples) / (total_samples ** 2) * torch.outer(diff, diff)
        self.num_samples = total_samples

    def generate(self, num_samples_to_generate):
        """Generate a feature vector by sampling from the domain's distribution."""
        if self.num_samples == 0:
            raise ValueError("Cannot generate samples because the distribution is empty.")
        cov_reg = self.cov + 1e-4 * torch.eye(self.cov.shape[0], device=self.device)
        if torch.isnan(cov_reg).any() or torch.isinf(cov_reg).any():
            raise ValueError("Covariance matrix contains NaN or inf values after regularization.")
        mvn = dist.MultivariateNormal(self.mean, covariance_matrix=cov_reg)
        feature_vector = mvn.sample((num_samples_to_generate,)).to(self.device)

        return feature_vector
    

class MultiCentroidDist:
    def __init__(self, n_centroids, feature_dim, device):
        self.n_clusters = n_centroids
        self.feature_dim = feature_dim
        self.device = device
        self.cluster_means = None
        self.cluster_vars = None
        self.cluster_masks = None
    
    def compute_centroids(self, features:torch.Tensor):
        """Compute cluster means and variances using KMeans."""
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("Features contain NaN or inf values.")

        features_np = features.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters).fit(features_np)
        cluster_means = []
        cluster_vars = []
        cluster_masks = []
        km_labels = kmeans.labels_

        for i in range(self.n_clusters):
            cluster_mask = (km_labels == i)
            if cluster_mask.sum() == 0:
                # If no samples are assigned to this cluster, use the global mean and variance
                cluster_mean = np.mean(features_np, axis=0)
                cluster_var = np.var(features_np, axis=0)
            else:
                cluster_mean = np.mean(features_np[cluster_mask], axis=0)
                cluster_var = np.var(features_np[cluster_mask], axis=0)

            cluster_means.append(torch.tensor(cluster_mean).to(self.device))
            cluster_vars.append(torch.tensor(cluster_var).to(self.device))
            cluster_masks.append(cluster_mask)

        self.cluster_means = cluster_means
        self.cluster_vars = cluster_vars
        self.cluster_masks = cluster_masks

        assert len(self.cluster_means) == self.n_clusters

    def get_means_vector(self):
        return torch.stack(self.cluster_means, dim=0).to(self.device)

    def update(self, features:torch.Tensor):
        """Compute cluster means and variances using KMeans."""
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("Features contain NaN or inf values.")

        features_np = features.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters).fit(features_np)
        cluster_means = []
        cluster_vars = []
        cluster_masks = []
        km_labels = kmeans.labels_

        for i in range(self.n_clusters):
            cluster_mask = (km_labels == i)
            if cluster_mask.sum() == 0:
                # If no samples are assigned to this cluster, use the global mean and variance
                cluster_mean = np.mean(features_np, axis=0)
                cluster_var = np.var(features_np, axis=0)
            else:
                cluster_mean = np.mean(features_np[cluster_mask], axis=0)
                cluster_var = np.var(features_np[cluster_mask], axis=0)

            cluster_means.append(torch.tensor(cluster_mean).to(self.device))
            cluster_vars.append(torch.tensor(cluster_var).to(self.device))
            cluster_masks.append(cluster_mask)

        self.cluster_means.extend(cluster_means)
        self.cluster_vars.extend(cluster_vars)
        self.cluster_masks.extend(cluster_masks)

        self.n_clusters += len(cluster_means)

        assert len(self.cluster_means) == self.n_clusters

    def closest_id(self, features):
        closest_indices = []
        for i in range(self.n_clusters):
            cluster_mean = self.cluster_means[i]
            dist = (features - cluster_mean).pow(2).sum(dim=1).sqrt()
            min_idx = dist.argmin()
            closest_indices.append(min_idx)
        closest_indices = torch.tensor(closest_indices).to(self.device)
        return closest_indices
    
    def generate(self, num_samples_to_generate):
        """Generate a feature vector by sampling from the domain's distribution."""
        if self.cluster_means is None:
            raise ValueError("Cannot generate samples because the centroids are not computed.")
        else:
            feature_vectors = []
            for i in range(self.n_clusters):
                # if self.cluster_vars[i].sum() == 0:
                #     continue
                cov_reg = torch.diag(self.cluster_vars[i]) + 1e-4 * torch.eye(self.feature_dim, device=self.device)
                # Check for invalid values
                if torch.isnan(cov_reg).any() or torch.isinf(cov_reg).any():
                    raise ValueError(f"Covariance matrix for cluster {i} contains NaN or inf values.")
                mvn = dist.MultivariateNormal(
                    self.cluster_means[i], 
                    covariance_matrix=cov_reg
                )
                feature_vector = mvn.sample((num_samples_to_generate,))
                feature_vectors.append(feature_vector)
            feature_vectors = torch.cat(feature_vectors, dim=0).to(self.device)
            shuffle_idx = torch.randperm(feature_vectors.shape[0])
            feature_vectors = feature_vectors[shuffle_idx[:num_samples_to_generate]]
        
        return feature_vectors
    
    def generate_per_centroid(self, num_samples_per_centroid):
        """Generate a feature vector by sampling from the domain's distribution."""
        if self.cluster_means is None:
            raise ValueError("Cannot generate samples because the centroids are not computed.")
        else:
            feature_vectors = []
            for i in range(self.n_clusters):
                # if self.cluster_vars[i].sum() == 0:
                #     continue
                cov_reg = torch.diag(self.cluster_vars[i]) + 1e-4 * torch.eye(self.feature_dim, device=self.device)
                # Check for invalid values
                if torch.isnan(cov_reg).any() or torch.isinf(cov_reg).any():
                    raise ValueError(f"Covariance matrix for cluster {i} contains NaN or inf values.")
                mvn = dist.MultivariateNormal(
                    self.cluster_means[i], 
                    covariance_matrix=cov_reg
                )
                feature_vector = mvn.sample((num_samples_per_centroid,))
                feature_vectors.append(feature_vector)
            feature_vectors = torch.cat(feature_vectors, dim=0).to(self.device)
        
        return feature_vectors
    

class MultiPrototypeDist:
    def __init__(self, n_prototypes, feature_dim, device):
        self.n_prototypes = n_prototypes
        self.feature_dim = feature_dim
        self.device = device
        self.prototype_feats = None
        self.prototype_vars = None

    def init_from(self, prototype_ids, cluster_masks, features):
        """Update prototype features and variances."""
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("Features contain NaN or inf values.")
        prototype_feats = []
        prototype_vars = []
        for i in range(self.n_prototypes):
            prototype_feats.append(features[prototype_ids[i]])
            prototype_vars.append(np.var(features[cluster_masks[i]].cpu().numpy(), axis=0))
        self.prototype_feats = prototype_feats
        self.prototype_vars = prototype_vars

    def generate(self, num_samples_to_generate, shuffle_idx=None, decay=0.1):
        """Generate a feature vector by sampling from the domain's distribution."""
        if self.prototype_feats is None:
            raise ValueError("Cannot generate samples because the prototypes are not computed.")
        else:
            feature_vectors = []
            for i in range(self.n_prototypes):
                # Regularize the covariance matrix
                cov_reg = torch.diag(torch.tensor(self.prototype_vars[i], device=self.device)) + 1e-4 * torch.eye(self.feature_dim, device=self.device)

                # Check for invalid values
                if torch.isnan(cov_reg).any() or torch.isinf(cov_reg).any():
                    raise ValueError(f"Covariance matrix for prototype {i} contains NaN or inf values.")

                # Generate samples
                prototype_feat = self.prototype_feats[i] * (0.9 + decay)
                mvn = dist.MultivariateNormal(prototype_feat, covariance_matrix=cov_reg)
                feature_vector = mvn.sample((num_samples_to_generate,))
                feature_vectors.append(feature_vector)
            feature_vectors = torch.cat(feature_vectors, dim=0).to(self.device)
            if shuffle_idx is None:
                shuffle_idx = torch.randperm(feature_vectors.shape[0])
            feature_vectors = feature_vectors[shuffle_idx[:num_samples_to_generate]]
        
        return feature_vectors, shuffle_idx

    def generate_per_prototype(self, num_samples_per_prototype, decay=0.1):
        """Generate a feature vector by sampling from the domain's distribution."""
        if self.prototype_feats is None:
            raise ValueError("Cannot generate samples because the prototypes are not computed.")
        else:
            feature_vectors = []
            for i in range(self.n_prototypes):
                # Regularize the covariance matrix
                cov_reg = torch.diag(torch.tensor(self.prototype_vars[i], device=self.device)) + 1e-4 * torch.eye(self.feature_dim, device=self.device)

                # Check for invalid values
                if torch.isnan(cov_reg).any() or torch.isinf(cov_reg).any():
                    raise ValueError(f"Covariance matrix for prototype {i} contains NaN or inf values.")

                # Generate samples
                prototype_feat = self.prototype_feats[i] * (0.9 + decay)
                mvn = dist.MultivariateNormal(prototype_feat, covariance_matrix=cov_reg)
                feature_vector = mvn.sample((num_samples_per_prototype,))
                feature_vectors.append(feature_vector)
            feature_vectors = torch.cat(feature_vectors, dim=0).to(self.device)
        
        return feature_vectors
    
    def get_prototype_vectors(self):
        return torch.stack(self.prototype_feats, dim=0).to(self.device)
