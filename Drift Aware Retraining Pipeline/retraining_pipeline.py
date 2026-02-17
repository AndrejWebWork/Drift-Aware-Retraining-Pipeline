"""
Automated Retraining Pipeline
Incremental learning, catastrophic forgetting prevention, versioned artifacts
"""
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import pickle
import hashlib
from datetime import datetime
import copy

@dataclass
class ModelVersion:
    version_id: str
    timestamp: datetime
    model_state: Any
    performance_metrics: Dict[str, float]
    training_config: Dict
    data_hash: str

class ElasticWeightConsolidation:
    """EWC: Prevents catastrophic forgetting by protecting important weights"""
    def __init__(self, lambda_ewc: float = 0.4):
        self.lambda_ewc = lambda_ewc
        self.fisher_information = {}
        self.optimal_params = {}
    
    def compute_fisher(self, model: Any, data: np.ndarray, labels: np.ndarray, 
                      get_gradients_fn: Callable) -> Dict:
        """Compute Fisher Information Matrix (diagonal approximation)"""
        fisher = {}
        
        # Get gradients for each sample
        gradients_list = []
        for i in range(min(len(data), 1000)):  # Sample for efficiency
            grads = get_gradients_fn(model, data[i:i+1], labels[i:i+1])
            gradients_list.append(grads)
        
        # Compute diagonal Fisher as squared gradient mean
        for param_name in gradients_list[0].keys():
            param_grads = [g[param_name] for g in gradients_list]
            fisher[param_name] = np.mean([g**2 for g in param_grads], axis=0)
        
        return fisher
    
    def store_optimal_params(self, model: Any, get_params_fn: Callable):
        """Store current model parameters as optimal"""
        self.optimal_params = get_params_fn(model)
    
    def compute_ewc_loss(self, model: Any, get_params_fn: Callable) -> float:
        """Compute EWC regularization loss"""
        if not self.fisher_information or not self.optimal_params:
            return 0.0
        
        current_params = get_params_fn(model)
        ewc_loss = 0.0
        
        for param_name in self.fisher_information.keys():
            if param_name in current_params and param_name in self.optimal_params:
                fisher = self.fisher_information[param_name]
                optimal = self.optimal_params[param_name]
                current = current_params[param_name]
                
                ewc_loss += np.sum(fisher * (current - optimal)**2)
        
        return self.lambda_ewc * ewc_loss / 2

class LearningWithoutForgetting:
    """LwF: Knowledge distillation from old model to new model"""
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha
        self.old_model = None
    
    def store_old_model(self, model: Any):
        """Store copy of old model"""
        self.old_model = copy.deepcopy(model)
    
    def compute_distillation_loss(self, new_predictions: np.ndarray, 
                                 old_predictions: np.ndarray) -> float:
        """Compute knowledge distillation loss"""
        if self.old_model is None:
            return 0.0
        
        # Soften predictions with temperature
        new_soft = self._soften_predictions(new_predictions)
        old_soft = self._soften_predictions(old_predictions)
        
        # KL divergence
        kl_div = np.sum(old_soft * np.log((old_soft + 1e-10) / (new_soft + 1e-10)))
        
        return self.alpha * kl_div * (self.temperature ** 2)
    
    def _soften_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Apply temperature softening"""
        exp_pred = np.exp(predictions / self.temperature)
        return exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)

class IncrementalTrainer:
    """Handles incremental and warm-start training"""
    def __init__(self, use_ewc: bool = True, use_lwf: bool = True):
        self.use_ewc = use_ewc
        self.use_lwf = use_lwf
        self.ewc = ElasticWeightConsolidation() if use_ewc else None
        self.lwf = LearningWithoutForgetting() if use_lwf else None
    
    def prepare_incremental_training(self, old_model: Any, old_data: np.ndarray, 
                                    old_labels: np.ndarray,
                                    get_gradients_fn: Callable,
                                    get_params_fn: Callable):
        """Prepare for incremental training by computing Fisher and storing old model"""
        if self.use_ewc and self.ewc:
            self.ewc.fisher_information = self.ewc.compute_fisher(
                old_model, old_data, old_labels, get_gradients_fn
            )
            self.ewc.store_optimal_params(old_model, get_params_fn)
        
        if self.use_lwf and self.lwf:
            self.lwf.store_old_model(old_model)
    
    def compute_regularization_loss(self, model: Any, new_predictions: np.ndarray,
                                   old_predictions: np.ndarray,
                                   get_params_fn: Callable) -> float:
        """Compute combined regularization loss"""
        total_loss = 0.0
        
        if self.use_ewc and self.ewc:
            total_loss += self.ewc.compute_ewc_loss(model, get_params_fn)
        
        if self.use_lwf and self.lwf:
            total_loss += self.lwf.compute_distillation_loss(new_predictions, old_predictions)
        
        return total_loss

class DataCurator:
    """Curates and filters training data"""
    def __init__(self, min_confidence: float = 0.6, max_samples: int = 50000):
        self.min_confidence = min_confidence
        self.max_samples = max_samples
    
    def curate(self, data: np.ndarray, labels: np.ndarray, 
              confidences: Optional[np.ndarray] = None,
              timestamps: Optional[np.ndarray] = None) -> tuple:
        """Filter and select high-quality training data"""
        
        # Filter by confidence if available
        if confidences is not None:
            mask = confidences >= self.min_confidence
            data = data[mask]
            labels = labels[mask]
            if timestamps is not None:
                timestamps = timestamps[mask]
        
        # Prioritize recent data if timestamps available
        if timestamps is not None and len(data) > self.max_samples:
            sorted_indices = np.argsort(timestamps)[-self.max_samples:]
            data = data[sorted_indices]
            labels = labels[sorted_indices]
        elif len(data) > self.max_samples:
            # Random sampling
            indices = np.random.choice(len(data), self.max_samples, replace=False)
            data = data[indices]
            labels = labels[indices]
        
        return data, labels
    
    def balance_classes(self, data: np.ndarray, labels: np.ndarray) -> tuple:
        """Balance class distribution"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if len(unique_labels) <= 1:
            return data, labels
        
        min_count = min(counts)
        balanced_data = []
        balanced_labels = []
        
        for label in unique_labels:
            label_mask = labels == label
            label_data = data[label_mask]
            label_labels = labels[label_mask]
            
            if len(label_data) > min_count:
                indices = np.random.choice(len(label_data), min_count, replace=False)
                label_data = label_data[indices]
                label_labels = label_labels[indices]
            
            balanced_data.append(label_data)
            balanced_labels.append(label_labels)
        
        return np.vstack(balanced_data), np.concatenate(balanced_labels)

class ModelRegistry:
    """Manages model versions and artifacts"""
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        self.versions = []
    
    def register_model(self, model: Any, performance_metrics: Dict[str, float],
                      training_config: Dict, training_data_hash: str) -> str:
        """Register new model version"""
        version_id = self._generate_version_id(model, training_data_hash)
        
        version = ModelVersion(
            version_id=version_id,
            timestamp=datetime.now(),
            model_state=self._serialize_model(model),
            performance_metrics=performance_metrics,
            training_config=training_config,
            data_hash=training_data_hash
        )
        
        self.versions.append(version)
        self._save_version(version)
        
        return version_id
    
    def get_model(self, version_id: str) -> Optional[Any]:
        """Retrieve model by version ID"""
        for version in self.versions:
            if version.version_id == version_id:
                return self._deserialize_model(version.model_state)
        return None
    
    def get_best_model(self, metric: str = 'accuracy') -> Optional[Any]:
        """Get best performing model"""
        if not self.versions:
            return None
        
        best_version = max(self.versions, 
                          key=lambda v: v.performance_metrics.get(metric, 0))
        return self._deserialize_model(best_version.model_state)
    
    def _generate_version_id(self, model: Any, data_hash: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().isoformat()
        content = f"{timestamp}_{data_hash}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _serialize_model(self, model: Any) -> bytes:
        """Serialize model to bytes"""
        return pickle.dumps(model)
    
    def _deserialize_model(self, model_bytes: bytes) -> Any:
        """Deserialize model from bytes"""
        return pickle.loads(model_bytes)
    
    def _save_version(self, version: ModelVersion):
        """Persist version to disk"""
        import os
        os.makedirs(self.registry_path, exist_ok=True)
        filepath = f"{self.registry_path}/{version.version_id}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(version, f)

class RetrainingOrchestrator:
    """Main orchestrator for automated retraining"""
    def __init__(self, registry_path: str = "./model_registry"):
        self.incremental_trainer = IncrementalTrainer()
        self.data_curator = DataCurator()
        self.model_registry = ModelRegistry(registry_path)
    
    def execute_retraining(self,
                          current_model: Any,
                          new_data: np.ndarray,
                          new_labels: np.ndarray,
                          retraining_policy: Any,
                          train_fn: Callable,
                          get_gradients_fn: Callable,
                          get_params_fn: Callable,
                          old_data: Optional[np.ndarray] = None,
                          old_labels: Optional[np.ndarray] = None) -> Any:
        """Execute retraining based on policy"""
        
        # Curate data
        curated_data, curated_labels = self.data_curator.curate(new_data, new_labels)
        
        # Balance classes
        curated_data, curated_labels = self.data_curator.balance_classes(
            curated_data, curated_labels
        )
        
        # Prepare incremental training if needed
        if retraining_policy.decision.value in ['partial_retrain', 'incremental_update']:
            if old_data is not None and old_labels is not None:
                self.incremental_trainer.prepare_incremental_training(
                    current_model, old_data, old_labels,
                    get_gradients_fn, get_params_fn
                )
        
        # Train new model
        new_model = train_fn(
            current_model,
            curated_data,
            curated_labels,
            retraining_policy,
            self.incremental_trainer
        )
        
        return new_model
    
    def register_trained_model(self, model: Any, metrics: Dict[str, float],
                              config: Dict, data: np.ndarray) -> str:
        """Register newly trained model"""
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
        return self.model_registry.register_model(model, metrics, config, data_hash)
