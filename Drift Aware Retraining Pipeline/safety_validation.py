"""
Performance Safety & Validation Layer
Champion-Challenger testing, statistical significance, automatic rollback
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ValidationResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"

@dataclass
class SafetyReport:
    validation_result: ValidationResult
    champion_performance: Dict[str, float]
    challenger_performance: Dict[str, float]
    statistical_significance: Dict[str, Tuple[float, float]]
    regression_detected: bool
    recommendation: str
    details: Dict

class ShadowEvaluator:
    """Evaluates challenger model in shadow mode"""
    def __init__(self, min_samples: int = 1000):
        self.min_samples = min_samples
        self.champion_predictions = []
        self.challenger_predictions = []
        self.ground_truth = []
    
    def add_prediction(self, champion_pred: np.ndarray, 
                      challenger_pred: np.ndarray,
                      true_label: Optional[np.ndarray] = None):
        """Collect predictions from both models"""
        self.champion_predictions.append(champion_pred)
        self.challenger_predictions.append(challenger_pred)
        if true_label is not None:
            self.ground_truth.append(true_label)
    
    def evaluate(self, metrics_fn: callable) -> Tuple[Dict, Dict]:
        """Evaluate both models on collected data"""
        if len(self.champion_predictions) < self.min_samples:
            return {}, {}
        
        champion_preds = np.array(self.champion_predictions)
        challenger_preds = np.array(self.challenger_predictions)
        ground_truth = np.array(self.ground_truth) if self.ground_truth else None
        
        champion_metrics = metrics_fn(champion_preds, ground_truth)
        challenger_metrics = metrics_fn(challenger_preds, ground_truth)
        
        return champion_metrics, challenger_metrics
    
    def reset(self):
        """Clear collected predictions"""
        self.champion_predictions = []
        self.challenger_predictions = []
        self.ground_truth = []

class StatisticalSignificanceTester:
    """Tests statistical significance of performance differences"""
    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.01):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
    
    def permutation_test(self, champion_scores: np.ndarray, 
                        challenger_scores: np.ndarray,
                        n_permutations: int = 10000) -> Tuple[float, float]:
        """Permutation test for comparing two models"""
        observed_diff = np.mean(challenger_scores) - np.mean(champion_scores)
        
        combined = np.concatenate([champion_scores, challenger_scores])
        n_champion = len(champion_scores)
        
        permuted_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_champion = combined[:n_champion]
            perm_challenger = combined[n_champion:]
            perm_diff = np.mean(perm_challenger) - np.mean(perm_champion)
            permuted_diffs.append(perm_diff)
        
        permuted_diffs = np.array(permuted_diffs)
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        return observed_diff, p_value
    
    def paired_t_test(self, champion_scores: np.ndarray,
                     challenger_scores: np.ndarray) -> Tuple[float, float]:
        """Paired t-test for matched samples"""
        if len(champion_scores) != len(challenger_scores):
            raise ValueError("Scores must have same length for paired test")
        
        differences = challenger_scores - champion_scores
        t_stat, p_value = stats.ttest_1samp(differences, 0)
        
        return float(np.mean(differences)), p_value
    
    def mcnemar_test(self, champion_correct: np.ndarray,
                    challenger_correct: np.ndarray) -> Tuple[float, float]:
        """McNemar's test for classification models"""
        # Contingency table
        both_correct = np.sum(champion_correct & challenger_correct)
        both_wrong = np.sum(~champion_correct & ~challenger_correct)
        champion_only = np.sum(champion_correct & ~challenger_correct)
        challenger_only = np.sum(~champion_correct & challenger_correct)
        
        # McNemar statistic
        if champion_only + challenger_only == 0:
            return 0.0, 1.0
        
        statistic = (abs(champion_only - challenger_only) - 1)**2 / (champion_only + challenger_only)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        improvement = (challenger_only - champion_only) / len(champion_correct)
        
        return float(improvement), p_value
    
    def bootstrap_confidence_interval(self, scores: np.ndarray,
                                     n_bootstrap: int = 10000,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval for metric"""
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return float(ci_lower), float(ci_upper)

class RegressionDetector:
    """Detects performance regressions"""
    def __init__(self, regression_threshold: float = 0.02, 
                 critical_metrics: List[str] = None):
        self.regression_threshold = regression_threshold
        self.critical_metrics = critical_metrics or ['accuracy', 'f1_score', 'auc']
    
    def detect(self, champion_metrics: Dict[str, float],
              challenger_metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Detect if challenger has regressed on critical metrics"""
        regressions = []
        
        for metric in self.critical_metrics:
            if metric not in champion_metrics or metric not in challenger_metrics:
                continue
            
            champion_value = champion_metrics[metric]
            challenger_value = challenger_metrics[metric]
            
            # Check for regression (assuming higher is better)
            if challenger_value < champion_value - self.regression_threshold:
                regressions.append(f"{metric}: {champion_value:.4f} -> {challenger_value:.4f}")
        
        return len(regressions) > 0, regressions

class ChampionChallengerValidator:
    """Main validator for champion-challenger testing"""
    def __init__(self, alpha: float = 0.05, min_samples: int = 1000):
        self.shadow_evaluator = ShadowEvaluator(min_samples)
        self.significance_tester = StatisticalSignificanceTester(alpha)
        self.regression_detector = RegressionDetector()
    
    def validate(self,
                champion_model: any,
                challenger_model: any,
                validation_data: np.ndarray,
                validation_labels: np.ndarray,
                metrics_fn: callable,
                predict_fn: callable) -> SafetyReport:
        """Comprehensive validation of challenger model"""
        
        # Get predictions
        champion_preds = predict_fn(champion_model, validation_data)
        challenger_preds = predict_fn(challenger_model, validation_data)
        
        # Compute metrics
        champion_metrics = metrics_fn(champion_preds, validation_labels)
        challenger_metrics = metrics_fn(challenger_preds, validation_labels)
        
        # Statistical significance testing
        significance_results = {}
        
        # For classification: use McNemar's test
        if len(champion_preds.shape) == 1 or champion_preds.shape[1] == 1:
            champion_correct = (champion_preds.flatten() == validation_labels.flatten())
            challenger_correct = (challenger_preds.flatten() == validation_labels.flatten())
            
            improvement, p_value = self.significance_tester.mcnemar_test(
                champion_correct, challenger_correct
            )
            significance_results['mcnemar'] = (improvement, p_value)
        
        # Permutation test on primary metric
        primary_metric = 'accuracy' if 'accuracy' in champion_metrics else list(champion_metrics.keys())[0]
        
        # Compute per-sample scores for permutation test
        champion_scores = self._compute_per_sample_scores(
            champion_preds, validation_labels, primary_metric
        )
        challenger_scores = self._compute_per_sample_scores(
            challenger_preds, validation_labels, primary_metric
        )
        
        diff, p_value = self.significance_tester.permutation_test(
            champion_scores, challenger_scores
        )
        significance_results['permutation'] = (diff, p_value)
        
        # Bootstrap confidence intervals
        champion_ci = self.significance_tester.bootstrap_confidence_interval(champion_scores)
        challenger_ci = self.significance_tester.bootstrap_confidence_interval(challenger_scores)
        
        # Regression detection
        regression_detected, regression_details = self.regression_detector.detect(
            champion_metrics, challenger_metrics
        )
        
        # Make decision
        is_significant = any(p < self.significance_tester.alpha 
                           for _, p in significance_results.values())
        is_improvement = challenger_metrics.get(primary_metric, 0) > champion_metrics.get(primary_metric, 0)
        
        if regression_detected:
            result = ValidationResult.FAIL
            recommendation = "REJECT challenger - performance regression detected"
        elif is_significant and is_improvement:
            result = ValidationResult.PASS
            recommendation = "ACCEPT challenger - statistically significant improvement"
        elif is_improvement and not is_significant:
            result = ValidationResult.INCONCLUSIVE
            recommendation = "MONITOR challenger - improvement not statistically significant, collect more data"
        else:
            result = ValidationResult.FAIL
            recommendation = "REJECT challenger - no significant improvement"
        
        return SafetyReport(
            validation_result=result,
            champion_performance=champion_metrics,
            challenger_performance=challenger_metrics,
            statistical_significance=significance_results,
            regression_detected=regression_detected,
            recommendation=recommendation,
            details={
                'regression_details': regression_details,
                'champion_ci': champion_ci,
                'challenger_ci': challenger_ci,
                'sample_size': len(validation_data)
            }
        )
    
    def _compute_per_sample_scores(self, predictions: np.ndarray, 
                                   labels: np.ndarray, 
                                   metric: str) -> np.ndarray:
        """Compute per-sample scores for statistical testing"""
        if metric == 'accuracy':
            return (predictions.flatten() == labels.flatten()).astype(float)
        else:
            # For other metrics, use binary correctness as proxy
            return (predictions.flatten() == labels.flatten()).astype(float)

class AutomaticRollback:
    """Handles automatic rollback on validation failure"""
    def __init__(self, model_registry: any):
        self.model_registry = model_registry
        self.rollback_history = []
    
    def rollback(self, reason: str, failed_version_id: str) -> any:
        """Rollback to previous stable version"""
        # Get previous best model
        champion_model = self.model_registry.get_best_model()
        
        self.rollback_history.append({
            'timestamp': np.datetime64('now'),
            'reason': reason,
            'failed_version': failed_version_id
        })
        
        return champion_model
    
    def can_rollback(self) -> bool:
        """Check if rollback is possible"""
        return len(self.model_registry.versions) > 1
