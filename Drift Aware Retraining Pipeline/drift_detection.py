"""
Drift Detection Module - Multi-Detector Ensemble
Zero-cost, production-grade drift detection
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class DriftType(Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    LABEL_DRIFT = "label_drift"

class DriftSeverity(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class DriftResult:
    detector_name: str
    drift_detected: bool
    drift_score: float
    p_value: Optional[float]
    severity: DriftSeverity
    confidence: float
    feature_name: Optional[str] = None
    metadata: Dict = None

class KSDetector:
    """Kolmogorov-Smirnov test for univariate drift"""
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def detect(self, reference: np.ndarray, current: np.ndarray, feature_name: str = None) -> DriftResult:
        statistic, p_value = ks_2samp(reference, current)
        drift_detected = p_value < self.alpha
        severity = self._score_to_severity(statistic)
        
        return DriftResult(
            detector_name="KS-Test",
            drift_detected=drift_detected,
            drift_score=statistic,
            p_value=p_value,
            severity=severity,
            confidence=1 - p_value,
            feature_name=feature_name
        )
    
    def _score_to_severity(self, score: float) -> DriftSeverity:
        if score < 0.1: return DriftSeverity.NONE
        if score < 0.2: return DriftSeverity.LOW
        if score < 0.3: return DriftSeverity.MEDIUM
        if score < 0.5: return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL

class PSIDetector:
    """Population Stability Index - industry standard"""
    def __init__(self, bins: int = 10, thresholds: Tuple[float, float, float] = (0.1, 0.2, 0.25)):
        self.bins = bins
        self.thresholds = thresholds
    
    def detect(self, reference: np.ndarray, current: np.ndarray, feature_name: str = None) -> DriftResult:
        ref_hist, bin_edges = np.histogram(reference, bins=self.bins)
        cur_hist, _ = np.histogram(current, bins=bin_edges)
        
        ref_pct = (ref_hist + 1e-10) / len(reference)
        cur_pct = (cur_hist + 1e-10) / len(current)
        
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        severity = self._psi_to_severity(psi)
        drift_detected = psi > self.thresholds[0]
        
        return DriftResult(
            detector_name="PSI",
            drift_detected=drift_detected,
            drift_score=psi,
            p_value=None,
            severity=severity,
            confidence=min(psi / self.thresholds[2], 1.0),
            feature_name=feature_name
        )
    
    def _psi_to_severity(self, psi: float) -> DriftSeverity:
        if psi < self.thresholds[0]: return DriftSeverity.NONE
        if psi < self.thresholds[1]: return DriftSeverity.LOW
        if psi < self.thresholds[2]: return DriftSeverity.MEDIUM
        return DriftSeverity.HIGH

class JSDetector:
    """Jensen-Shannon Divergence detector"""
    def __init__(self, bins: int = 50, threshold: float = 0.1):
        self.bins = bins
        self.threshold = threshold
    
    def detect(self, reference: np.ndarray, current: np.ndarray, feature_name: str = None) -> DriftResult:
        ref_hist, bin_edges = np.histogram(reference, bins=self.bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        ref_hist = ref_hist + 1e-10
        cur_hist = cur_hist + 1e-10
        ref_hist /= ref_hist.sum()
        cur_hist /= cur_hist.sum()
        
        js_distance = jensenshannon(ref_hist, cur_hist)
        drift_detected = js_distance > self.threshold
        severity = self._js_to_severity(js_distance)
        
        return DriftResult(
            detector_name="JS-Divergence",
            drift_detected=drift_detected,
            drift_score=float(js_distance),
            p_value=None,
            severity=severity,
            confidence=min(js_distance / 0.5, 1.0),
            feature_name=feature_name
        )
    
    def _js_to_severity(self, js: float) -> DriftSeverity:
        if js < 0.1: return DriftSeverity.NONE
        if js < 0.2: return DriftSeverity.LOW
        if js < 0.3: return DriftSeverity.MEDIUM
        if js < 0.5: return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL

class MMDDetector:
    """Maximum Mean Discrepancy with RBF kernel"""
    def __init__(self, gamma: float = 1.0, threshold: float = 0.05):
        self.gamma = gamma
        self.threshold = threshold
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
        K = np.exp(-self.gamma * (X_norm + Y_norm - 2 * X @ Y.T))
        return K
    
    def detect(self, reference: np.ndarray, current: np.ndarray, feature_name: str = None) -> DriftResult:
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if current.ndim == 1:
            current = current.reshape(-1, 1)
        
        K_xx = self._rbf_kernel(reference, reference)
        K_yy = self._rbf_kernel(current, current)
        K_xy = self._rbf_kernel(reference, current)
        
        mmd_squared = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        mmd = np.sqrt(max(mmd_squared, 0))
        
        drift_detected = mmd > self.threshold
        severity = self._mmd_to_severity(mmd)
        
        return DriftResult(
            detector_name="MMD",
            drift_detected=drift_detected,
            drift_score=float(mmd),
            p_value=None,
            severity=severity,
            confidence=min(mmd / (2 * self.threshold), 1.0),
            feature_name=feature_name
        )
    
    def _mmd_to_severity(self, mmd: float) -> DriftSeverity:
        if mmd < 0.05: return DriftSeverity.NONE
        if mmd < 0.1: return DriftSeverity.LOW
        if mmd < 0.2: return DriftSeverity.MEDIUM
        if mmd < 0.4: return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL

class ADWINDetector:
    """Adaptive Windowing for streaming data"""
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = []
        self.width = 0
    
    def add_element(self, value: float) -> Tuple[bool, float]:
        self.window.append(value)
        self.width += 1
        
        if self.width == 1:
            return False, 0.0
        
        drift_detected = False
        drift_score = 0.0
        
        for i in range(1, self.width):
            w0 = i
            w1 = self.width - i
            
            mean0 = sum(self.window[:i]) / w0
            mean1 = sum(self.window[i:]) / w1
            
            diff = abs(mean0 - mean1)
            m = 1 / (1/w0 + 1/w1)
            epsilon = np.sqrt(2 * m * np.log(2 / self.delta))
            
            if diff > epsilon:
                drift_detected = True
                drift_score = diff / epsilon
                self.window = self.window[i:]
                self.width = len(self.window)
                break
        
        return drift_detected, drift_score

class PredictionDriftDetector:
    """Detects drift in model predictions"""
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
    
    def detect(self, reference_preds: np.ndarray, current_preds: np.ndarray) -> DriftResult:
        if reference_preds.ndim == 2:
            ref_mean = reference_preds.mean(axis=0)
            cur_mean = current_preds.mean(axis=0)
            js_dist = jensenshannon(ref_mean + 1e-10, cur_mean + 1e-10)
            drift_score = float(js_dist)
        else:
            ks_stat, p_value = ks_2samp(reference_preds.flatten(), current_preds.flatten())
            drift_score = float(ks_stat)
        
        drift_detected = drift_score > self.threshold
        severity = self._score_to_severity(drift_score)
        
        return DriftResult(
            detector_name="Prediction-Drift",
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=None,
            severity=severity,
            confidence=min(drift_score / (2 * self.threshold), 1.0)
        )
    
    def _score_to_severity(self, score: float) -> DriftSeverity:
        if score < 0.1: return DriftSeverity.NONE
        if score < 0.2: return DriftSeverity.LOW
        if score < 0.3: return DriftSeverity.MEDIUM
        if score < 0.5: return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL

class DriftEnsemble:
    """Aggregates multiple drift detectors with Bayesian voting"""
    def __init__(self):
        self.detectors = {
            'ks': KSDetector(),
            'psi': PSIDetector(),
            'js': JSDetector(),
            'mmd': MMDDetector()
        }
    
    def detect_feature_drift(self, reference: np.ndarray, current: np.ndarray, 
                            feature_name: str = None) -> Dict[str, DriftResult]:
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.detect(reference, current, feature_name)
        return results
    
    def aggregate_results(self, results: Dict[str, DriftResult]) -> DriftResult:
        """Bayesian ensemble aggregation"""
        drift_votes = sum(1 for r in results.values() if r.drift_detected)
        total_votes = len(results)
        
        weights = {'ks': 0.3, 'psi': 0.25, 'js': 0.25, 'mmd': 0.2}
        weighted_score = sum(weights.get(k, 1/total_votes) * r.drift_score 
                            for k, r in results.items())
        
        prior_drift_prob = 0.1
        likelihood_drift = drift_votes / total_votes
        posterior = (likelihood_drift * prior_drift_prob) / \
                   (likelihood_drift * prior_drift_prob + (1 - likelihood_drift) * (1 - prior_drift_prob))
        
        drift_detected = drift_votes >= (total_votes / 2)
        
        if weighted_score < 0.1: severity = DriftSeverity.NONE
        elif weighted_score < 0.2: severity = DriftSeverity.LOW
        elif weighted_score < 0.3: severity = DriftSeverity.MEDIUM
        elif weighted_score < 0.5: severity = DriftSeverity.HIGH
        else: severity = DriftSeverity.CRITICAL
        
        return DriftResult(
            detector_name="Ensemble",
            drift_detected=drift_detected,
            drift_score=weighted_score,
            p_value=None,
            severity=severity,
            confidence=posterior,
            metadata={'votes': drift_votes, 'total': total_votes, 'individual_results': results}
        )
