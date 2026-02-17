"""
Severity & Root-Cause Analysis Module
Bayesian severity scoring, feature attribution, temporal localization
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SeverityAnalysis:
    overall_severity: float
    confidence_interval: Tuple[float, float]
    feature_attributions: Dict[str, float]
    temporal_segments: List[Dict]
    root_cause_hypothesis: str
    recommended_action: str

class BayesianSeverityScorer:
    """Bayesian approach to drift severity quantification"""
    def __init__(self, prior_mean: float = 0.1, prior_std: float = 0.05):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
    
    def score(self, drift_scores: List[float], confidences: List[float]) -> Tuple[float, Tuple[float, float]]:
        """Compute posterior severity with confidence interval"""
        if not drift_scores:
            return 0.0, (0.0, 0.0)
        
        weighted_scores = np.array(drift_scores) * np.array(confidences)
        mean_score = weighted_scores.sum() / (sum(confidences) + 1e-10)
        
        # Bayesian update
        likelihood_precision = 1 / (np.var(drift_scores) + 1e-10)
        prior_precision = 1 / (self.prior_std ** 2)
        
        posterior_precision = prior_precision + len(drift_scores) * likelihood_precision
        posterior_mean = (prior_precision * self.prior_mean + likelihood_precision * sum(drift_scores)) / posterior_precision
        posterior_std = np.sqrt(1 / posterior_precision)
        
        # 95% confidence interval
        ci_lower = max(0, posterior_mean - 1.96 * posterior_std)
        ci_upper = min(1, posterior_mean + 1.96 * posterior_std)
        
        return float(posterior_mean), (float(ci_lower), float(ci_upper))

class FeatureAttributor:
    """SHAP-inspired feature-level drift attribution"""
    def __init__(self):
        pass
    
    def attribute(self, feature_drift_scores: Dict[str, float], 
                 feature_importances: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Compute feature-level contribution to overall drift"""
        if not feature_drift_scores:
            return {}
        
        # Normalize drift scores
        total_drift = sum(feature_drift_scores.values())
        if total_drift == 0:
            return {k: 0.0 for k in feature_drift_scores}
        
        base_attributions = {k: v / total_drift for k, v in feature_drift_scores.items()}
        
        # Weight by feature importance if available
        if feature_importances:
            weighted_attributions = {}
            total_weighted = 0
            for feature, drift_attr in base_attributions.items():
                importance = feature_importances.get(feature, 1.0)
                weighted_attributions[feature] = drift_attr * importance
                total_weighted += weighted_attributions[feature]
            
            # Renormalize
            if total_weighted > 0:
                base_attributions = {k: v / total_weighted for k, v in weighted_attributions.items()}
        
        # Sort by attribution
        return dict(sorted(base_attributions.items(), key=lambda x: x[1], reverse=True))

class TemporalLocalizer:
    """Identifies when drift occurred in time series"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
    
    def localize(self, data_stream: np.ndarray, timestamps: Optional[List[datetime]] = None) -> List[Dict]:
        """Detect temporal segments with drift"""
        if len(data_stream) < 2 * self.window_size:
            return []
        
        segments = []
        n = len(data_stream)
        
        for i in range(0, n - self.window_size, self.window_size // 2):
            window1 = data_stream[max(0, i - self.window_size):i]
            window2 = data_stream[i:i + self.window_size]
            
            if len(window1) < 10 or len(window2) < 10:
                continue
            
            # KS test between windows
            from scipy.stats import ks_2samp
            stat, p_value = ks_2samp(window1, window2)
            
            if p_value < 0.05:
                segment = {
                    'start_idx': i,
                    'end_idx': i + self.window_size,
                    'drift_score': stat,
                    'p_value': p_value
                }
                
                if timestamps and i < len(timestamps):
                    segment['timestamp'] = timestamps[i]
                
                segments.append(segment)
        
        return segments

class RootCauseAnalyzer:
    """Hypothesize root causes of drift"""
    def __init__(self):
        self.drift_patterns = {
            'sudden': 'Sudden drift detected - possible data pipeline change, system update, or external shock',
            'gradual': 'Gradual drift detected - possible concept evolution, seasonal trend, or slow distribution shift',
            'recurring': 'Recurring drift detected - possible periodic pattern, seasonal effect, or cyclic behavior',
            'feature_specific': 'Feature-specific drift - possible upstream data source change or feature engineering issue'
        }
    
    def analyze(self, temporal_segments: List[Dict], feature_attributions: Dict[str, float],
                drift_history: Optional[List] = None) -> Tuple[str, str]:
        """Generate root cause hypothesis and recommended action"""
        
        # Detect drift pattern
        if not temporal_segments:
            pattern = 'no_drift'
            hypothesis = 'No significant drift detected'
            action = 'Continue monitoring'
        elif len(temporal_segments) == 1:
            pattern = 'sudden'
            hypothesis = self.drift_patterns['sudden']
            action = 'Investigate recent system changes; consider immediate retraining'
        elif len(temporal_segments) > 3:
            pattern = 'recurring'
            hypothesis = self.drift_patterns['recurring']
            action = 'Implement adaptive windowing or seasonal model; schedule periodic retraining'
        else:
            pattern = 'gradual'
            hypothesis = self.drift_patterns['gradual']
            action = 'Schedule incremental retraining; monitor trend continuation'
        
        # Check for feature-specific drift
        if feature_attributions:
            top_feature = max(feature_attributions, key=feature_attributions.get)
            top_attribution = feature_attributions[top_feature]
            
            if top_attribution > 0.5:
                hypothesis += f' | Primary driver: {top_feature} ({top_attribution:.2%} attribution)'
                action += f' | Prioritize investigation of {top_feature}'
        
        return hypothesis, action

class SeverityAnalyzer:
    """Main severity analysis orchestrator"""
    def __init__(self):
        self.bayesian_scorer = BayesianSeverityScorer()
        self.feature_attributor = FeatureAttributor()
        self.temporal_localizer = TemporalLocalizer()
        self.root_cause_analyzer = RootCauseAnalyzer()
    
    def analyze(self, 
                drift_results: Dict[str, any],
                feature_drift_scores: Dict[str, float],
                data_stream: Optional[np.ndarray] = None,
                timestamps: Optional[List[datetime]] = None,
                feature_importances: Optional[Dict[str, float]] = None) -> SeverityAnalysis:
        """Comprehensive severity analysis"""
        
        # Extract drift scores and confidences
        drift_scores = []
        confidences = []
        
        if 'individual_results' in drift_results.get('metadata', {}):
            for result in drift_results['metadata']['individual_results'].values():
                drift_scores.append(result.drift_score)
                confidences.append(result.confidence)
        else:
            drift_scores = [drift_results.get('drift_score', 0.0)]
            confidences = [drift_results.get('confidence', 0.5)]
        
        # Bayesian severity scoring
        overall_severity, confidence_interval = self.bayesian_scorer.score(drift_scores, confidences)
        
        # Feature attribution
        feature_attributions = self.feature_attributor.attribute(feature_drift_scores, feature_importances)
        
        # Temporal localization
        temporal_segments = []
        if data_stream is not None:
            temporal_segments = self.temporal_localizer.localize(data_stream, timestamps)
        
        # Root cause analysis
        hypothesis, action = self.root_cause_analyzer.analyze(
            temporal_segments, feature_attributions
        )
        
        return SeverityAnalysis(
            overall_severity=overall_severity,
            confidence_interval=confidence_interval,
            feature_attributions=feature_attributions,
            temporal_segments=temporal_segments,
            root_cause_hypothesis=hypothesis,
            recommended_action=action
        )
