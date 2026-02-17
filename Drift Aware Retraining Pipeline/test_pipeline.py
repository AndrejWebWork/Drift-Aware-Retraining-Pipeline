"""
Comprehensive Test Suite for Drift-Aware Retraining Pipeline
"""
import pytest
import numpy as np
from scipy import stats
import sys
sys.path.append('..')

from drift_detection import (
    KSDetector, PSIDetector, JSDetector, MMDDetector, 
    ADWINDetector, PredictionDriftDetector, DriftEnsemble,
    DriftSeverity
)
from severity_analysis import (
    BayesianSeverityScorer, FeatureAttributor, 
    TemporalLocalizer, SeverityAnalyzer
)
from decision_engine import (
    RuleBasedPolicy, RLBasedPolicy, DecisionEngine,
    RetrainingDecision
)

class TestDriftDetection:
    """Test drift detection components"""
    
    def test_ks_detector_no_drift(self):
        """Test KS detector with no drift"""
        detector = KSDetector(alpha=0.05)
        
        # Same distribution
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)
        
        result = detector.detect(ref, cur)
        
        assert result.detector_name == "KS-Test"
        assert result.p_value > 0.05  # Should not detect drift
        assert result.severity in [DriftSeverity.NONE, DriftSeverity.LOW]
    
    def test_ks_detector_with_drift(self):
        """Test KS detector with drift"""
        detector = KSDetector(alpha=0.05)
        
        # Different distributions
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(2, 1, 1000)  # Shifted mean
        
        result = detector.detect(ref, cur)
        
        assert result.drift_detected == True
        assert result.p_value < 0.05
        assert result.severity != DriftSeverity.NONE
    
    def test_psi_detector(self):
        """Test PSI detector"""
        detector = PSIDetector(bins=10)
        
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(1, 1, 1000)
        
        result = detector.detect(ref, cur)
        
        assert result.detector_name == "PSI"
        assert result.drift_score >= 0
        assert result.severity in DriftSeverity
    
    def test_js_detector(self):
        """Test Jensen-Shannon detector"""
        detector = JSDetector(bins=50)
        
        ref = np.random.uniform(0, 1, 1000)
        cur = np.random.uniform(0.5, 1.5, 1000)
        
        result = detector.detect(ref, cur)
        
        assert result.detector_name == "JS-Divergence"
        assert 0 <= result.drift_score <= 1
    
    def test_mmd_detector(self):
        """Test MMD detector"""
        detector = MMDDetector(gamma=1.0)
        
        ref = np.random.normal(0, 1, (500, 5))
        cur = np.random.normal(0.5, 1, (500, 5))
        
        result = detector.detect(ref, cur)
        
        assert result.detector_name == "MMD"
        assert result.drift_score >= 0
    
    def test_adwin_detector(self):
        """Test ADWIN streaming detector"""
        detector = ADWINDetector(delta=0.002)
        
        # Add stable data
        for i in range(100):
            drift, score = detector.add_element(np.random.normal(0, 1))
            assert drift == False
        
        # Add drifted data
        drift_detected = False
        for i in range(100):
            drift, score = detector.add_element(np.random.normal(5, 1))
            if drift:
                drift_detected = True
                break
        
        assert drift_detected == True
    
    def test_prediction_drift_detector(self):
        """Test prediction drift detector"""
        detector = PredictionDriftDetector(threshold=0.15)
        
        ref_preds = np.random.binomial(1, 0.7, 1000)
        cur_preds = np.random.binomial(1, 0.3, 1000)
        
        result = detector.detect(ref_preds, cur_preds)
        
        assert result.detector_name == "Prediction-Drift"
        assert result.drift_detected == True
    
    def test_drift_ensemble(self):
        """Test drift ensemble aggregation"""
        ensemble = DriftEnsemble()
        
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(2, 1, 1000)
        
        results = ensemble.detect_feature_drift(ref, cur, "test_feature")
        
        assert len(results) == 4  # KS, PSI, JS, MMD
        assert all(k in results for k in ['ks', 'psi', 'js', 'mmd'])
        
        aggregated = ensemble.aggregate_results(results)
        
        assert aggregated.detector_name == "Ensemble"
        assert 0 <= aggregated.confidence <= 1
        assert aggregated.metadata is not None

class TestSeverityAnalysis:
    """Test severity analysis components"""
    
    def test_bayesian_severity_scorer(self):
        """Test Bayesian severity scoring"""
        scorer = BayesianSeverityScorer()
        
        drift_scores = [0.3, 0.4, 0.35, 0.38]
        confidences = [0.8, 0.7, 0.75, 0.85]
        
        severity, ci = scorer.score(drift_scores, confidences)
        
        assert 0 <= severity <= 1
        assert ci[0] <= severity <= ci[1]
        assert ci[0] >= 0 and ci[1] <= 1
    
    def test_feature_attributor(self):
        """Test feature attribution"""
        attributor = FeatureAttributor()
        
        feature_scores = {
            'feature_0': 0.5,
            'feature_1': 0.3,
            'feature_2': 0.1,
            'feature_3': 0.05
        }
        
        attributions = attributor.attribute(feature_scores)
        
        assert len(attributions) == 4
        assert abs(sum(attributions.values()) - 1.0) < 0.01  # Should sum to 1
        assert list(attributions.keys())[0] == 'feature_0'  # Highest first
    
    def test_temporal_localizer(self):
        """Test temporal drift localization"""
        localizer = TemporalLocalizer(window_size=100)
        
        # Create data with drift at position 500
        data = np.concatenate([
            np.random.normal(0, 1, 500),
            np.random.normal(2, 1, 500)
        ])
        
        segments = localizer.localize(data)
        
        assert len(segments) > 0
        assert all('start_idx' in seg for seg in segments)
        assert all('drift_score' in seg for seg in segments)
    
    def test_severity_analyzer_integration(self):
        """Test full severity analyzer"""
        analyzer = SeverityAnalyzer()
        
        drift_results = {
            'drift_score': 0.4,
            'confidence': 0.8,
            'metadata': {}
        }
        
        feature_scores = {
            'feature_0': 0.5,
            'feature_1': 0.3
        }
        
        analysis = analyzer.analyze(
            drift_results=drift_results,
            feature_drift_scores=feature_scores
        )
        
        assert 0 <= analysis.overall_severity <= 1
        assert len(analysis.confidence_interval) == 2
        assert len(analysis.feature_attributions) > 0
        assert analysis.root_cause_hypothesis is not None
        assert analysis.recommended_action is not None

class TestDecisionEngine:
    """Test decision engine components"""
    
    def test_rule_based_policy_no_retrain(self):
        """Test rule-based policy - no retrain"""
        policy = RuleBasedPolicy()
        
        decision = policy.decide(
            severity=0.05,
            confidence=0.6,
            available_samples=2000
        )
        
        assert decision.decision == RetrainingDecision.NO_RETRAIN
        assert decision.estimated_cost == 0.0
    
    def test_rule_based_policy_full_retrain(self):
        """Test rule-based policy - full retrain"""
        policy = RuleBasedPolicy()
        
        decision = policy.decide(
            severity=0.6,
            confidence=0.8,
            available_samples=5000
        )
        
        assert decision.decision == RetrainingDecision.FULL_RETRAIN
        assert decision.estimated_cost > 0
        assert 'all' in decision.components_to_update
    
    def test_rule_based_policy_partial_retrain(self):
        """Test rule-based policy - partial retrain"""
        policy = RuleBasedPolicy()
        
        decision = policy.decide(
            severity=0.35,
            confidence=0.7,
            available_samples=3000,
            feature_attributions={'feature_0': 0.8, 'feature_1': 0.2}
        )
        
        assert decision.decision == RetrainingDecision.PARTIAL_RETRAIN
        assert len(decision.components_to_update) > 0
    
    def test_rule_based_policy_insufficient_samples(self):
        """Test rule-based policy - insufficient samples"""
        policy = RuleBasedPolicy(min_samples=1000)
        
        decision = policy.decide(
            severity=0.8,
            confidence=0.9,
            available_samples=500
        )
        
        assert decision.decision == RetrainingDecision.NO_RETRAIN
        assert "Insufficient samples" in decision.reasoning
    
    def test_rl_policy_initialization(self):
        """Test RL policy initialization"""
        policy = RLBasedPolicy(learning_rate=0.1, epsilon=0.1)
        
        assert policy.lr == 0.1
        assert policy.epsilon == 0.1
        assert len(policy.q_table) == 0
    
    def test_rl_policy_decision(self):
        """Test RL policy decision making"""
        policy = RLBasedPolicy(epsilon=0.0)  # No exploration
        
        # Initialize Q-table
        state = "5_8_2"
        policy.q_table[state] = {
            RetrainingDecision.NO_RETRAIN: 0.1,
            RetrainingDecision.INCREMENTAL_UPDATE: 0.3,
            RetrainingDecision.PARTIAL_RETRAIN: 0.5,
            RetrainingDecision.FULL_RETRAIN: 0.2
        }
        
        decision = policy.decide(severity=0.5, confidence=0.8, days_since_retrain=14)
        
        assert decision == RetrainingDecision.PARTIAL_RETRAIN  # Highest Q-value
    
    def test_rl_policy_update(self):
        """Test RL policy Q-value update"""
        policy = RLBasedPolicy(learning_rate=0.5, epsilon=0.0)
        
        # Make two decisions
        policy.decide(0.5, 0.8, 7)
        policy.decide(0.6, 0.7, 8)
        
        # Provide reward
        policy.update(reward=0.5)
        
        assert len(policy.reward_history) == 1
        assert policy.reward_history[0] == 0.5

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_drift_detection(self):
        """Test end-to-end drift detection flow"""
        # Create data with drift
        ref_data = np.random.normal(0, 1, (1000, 10))
        cur_data = np.random.normal(1, 1, (1000, 10))
        
        # Detect drift
        ensemble = DriftEnsemble()
        results = ensemble.detect_feature_drift(ref_data[:, 0], cur_data[:, 0])
        aggregated = ensemble.aggregate_results(results)
        
        # Analyze severity
        analyzer = SeverityAnalyzer()
        severity_analysis = analyzer.analyze(
            drift_results=aggregated.__dict__,
            feature_drift_scores={'feature_0': aggregated.drift_score}
        )
        
        # Make decision
        engine = DecisionEngine(use_rl=False)
        
        # Create mock severity analysis object
        class MockSeverityAnalysis:
            def __init__(self, severity):
                self.overall_severity = severity
                self.confidence_interval = (severity - 0.1, severity + 0.1)
                self.feature_attributions = {'feature_0': 1.0}
                self.temporal_segments = []
        
        mock_analysis = MockSeverityAnalysis(severity_analysis.overall_severity)
        
        policy = engine.decide(
            severity_analysis=mock_analysis,
            available_samples=1000
        )
        
        # Verify flow
        assert aggregated.drift_detected == True
        assert severity_analysis.overall_severity > 0
        assert policy.decision != RetrainingDecision.NO_RETRAIN

def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])

if __name__ == "__main__":
    run_tests()
