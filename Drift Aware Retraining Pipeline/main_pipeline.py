"""
Main Drift-Aware Retraining Pipeline Orchestrator
Integrates all components into production-ready system
"""
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from drift_detection import DriftEnsemble, PredictionDriftDetector, DriftType
from severity_analysis import SeverityAnalyzer
from decision_engine import DecisionEngine, RetrainingDecision
from retraining_pipeline import RetrainingOrchestrator
from safety_validation import ChampionChallengerValidator, AutomaticRollback, ValidationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    drift_detection_window: int = 1000
    min_samples_for_retrain: int = 1000
    validation_split: float = 0.2
    enable_rl_policy: bool = False
    enable_shadow_mode: bool = True
    max_retraining_frequency_days: int = 7
    registry_path: str = "./model_registry"

@dataclass
class PipelineState:
    current_model_version: str
    last_retrain_timestamp: Optional[datetime]
    drift_detected_count: int
    retraining_count: int
    rollback_count: int
    performance_history: List[Dict]

class DriftAwareRetrainingPipeline:
    """Main pipeline orchestrator - production-ready"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.drift_ensemble = DriftEnsemble()
        self.prediction_drift_detector = PredictionDriftDetector()
        self.severity_analyzer = SeverityAnalyzer()
        self.decision_engine = DecisionEngine(use_rl=self.config.enable_rl_policy)
        self.retraining_orchestrator = RetrainingOrchestrator(self.config.registry_path)
        self.validator = ChampionChallengerValidator()
        self.rollback_manager = AutomaticRollback(self.retraining_orchestrator.model_registry)
        
        # State
        self.state = PipelineState(
            current_model_version="initial",
            last_retrain_timestamp=None,
            drift_detected_count=0,
            retraining_count=0,
            rollback_count=0,
            performance_history=[]
        )
        
        # Data buffers
        self.reference_data = None
        self.reference_labels = None
        self.current_data_buffer = []
        self.current_labels_buffer = []
        self.current_predictions_buffer = []
        
        logger.info("Drift-Aware Retraining Pipeline initialized")
    
    def set_reference_data(self, data: np.ndarray, labels: np.ndarray):
        """Set reference data for drift detection"""
        self.reference_data = data
        self.reference_labels = labels
        logger.info(f"Reference data set: {len(data)} samples")
    
    def monitor_prediction(self, features: np.ndarray, prediction: np.ndarray, 
                          true_label: Optional[np.ndarray] = None):
        """Monitor single prediction for drift"""
        self.current_data_buffer.append(features)
        self.current_predictions_buffer.append(prediction)
        if true_label is not None:
            self.current_labels_buffer.append(true_label)
        
        # Check if buffer is full
        if len(self.current_data_buffer) >= self.config.drift_detection_window:
            self._check_drift_and_decide()
    
    def _check_drift_and_decide(self):
        """Check for drift and make retraining decision"""
        logger.info("Checking for drift...")
        
        current_data = np.array(self.current_data_buffer)
        current_predictions = np.array(self.current_predictions_buffer)
        
        # Detect drift on each feature
        feature_drift_results = {}
        feature_drift_scores = {}
        
        n_features = current_data.shape[1] if current_data.ndim > 1 else 1
        
        for i in range(min(n_features, 20)):  # Limit to 20 features for efficiency
            if current_data.ndim > 1:
                ref_feature = self.reference_data[:, i]
                cur_feature = current_data[:, i]
            else:
                ref_feature = self.reference_data
                cur_feature = current_data
            
            feature_results = self.drift_ensemble.detect_feature_drift(
                ref_feature, cur_feature, feature_name=f"feature_{i}"
            )
            
            aggregated = self.drift_ensemble.aggregate_results(feature_results)
            feature_drift_results[f"feature_{i}"] = aggregated
            feature_drift_scores[f"feature_{i}"] = aggregated.drift_score
        
        # Detect prediction drift
        if hasattr(self, 'reference_predictions'):
            pred_drift = self.prediction_drift_detector.detect(
                self.reference_predictions, current_predictions
            )
            feature_drift_results['predictions'] = pred_drift
            feature_drift_scores['predictions'] = pred_drift.drift_score
        
        # Aggregate all drift results
        all_results = list(feature_drift_results.values())
        if all_results:
            overall_drift = self.drift_ensemble.aggregate_results({
                f"detector_{i}": result for i, result in enumerate(all_results)
            })
            
            if overall_drift.drift_detected:
                self.state.drift_detected_count += 1
                logger.warning(f"Drift detected! Severity: {overall_drift.severity.name}, Score: {overall_drift.drift_score:.4f}")
                
                # Perform severity analysis
                severity_analysis = self.severity_analyzer.analyze(
                    drift_results=overall_drift.__dict__,
                    feature_drift_scores=feature_drift_scores,
                    data_stream=current_data.flatten() if current_data.ndim > 1 else current_data
                )
                
                logger.info(f"Severity Analysis: {severity_analysis.overall_severity:.4f}")
                logger.info(f"Root Cause: {severity_analysis.root_cause_hypothesis}")
                logger.info(f"Recommended Action: {severity_analysis.recommended_action}")
                
                # Make retraining decision
                days_since_retrain = None
                if self.state.last_retrain_timestamp:
                    days_since_retrain = (datetime.now() - self.state.last_retrain_timestamp).days
                
                policy = self.decision_engine.decide(
                    severity_analysis=severity_analysis,
                    available_samples=len(self.current_data_buffer),
                    days_since_last_retrain=days_since_retrain
                )
                
                logger.info(f"Retraining Decision: {policy.decision.value}")
                logger.info(f"Reasoning: {policy.reasoning}")
                
                # Execute retraining if decided
                if policy.decision != RetrainingDecision.NO_RETRAIN:
                    self._execute_retraining(policy, severity_analysis)
            else:
                logger.info("No significant drift detected")
        
        # Clear buffers
        self._reset_buffers()
    
    def _execute_retraining(self, policy: Any, severity_analysis: Any):
        """Execute retraining pipeline"""
        logger.info(f"Executing {policy.decision.value}...")
        
        # This is a placeholder - actual implementation would call user-provided training function
        logger.info("Retraining would be executed here with:")
        logger.info(f"  - Data window: [{policy.data_window_start}, {policy.data_window_end}]")
        logger.info(f"  - Components to update: {policy.components_to_update}")
        logger.info(f"  - Estimated cost: {policy.estimated_cost}")
        
        self.state.retraining_count += 1
        self.state.last_retrain_timestamp = datetime.now()
        
        # Note: Actual retraining would happen via user-provided callbacks
        # See execute_full_retraining_cycle() for complete implementation
    
    def _reset_buffers(self):
        """Reset data buffers"""
        self.current_data_buffer = []
        self.current_labels_buffer = []
        self.current_predictions_buffer = []
    
    def execute_full_retraining_cycle(self,
                                     current_model: Any,
                                     train_fn: Callable,
                                     predict_fn: Callable,
                                     metrics_fn: Callable,
                                     get_gradients_fn: Callable,
                                     get_params_fn: Callable,
                                     validation_data: np.ndarray,
                                     validation_labels: np.ndarray) -> Any:
        """Execute complete retraining cycle with validation"""
        
        logger.info("Starting full retraining cycle...")
        
        # Get current data
        new_data = np.array(self.current_data_buffer)
        new_labels = np.array(self.current_labels_buffer) if self.current_labels_buffer else None
        
        if new_labels is None or len(new_labels) == 0:
            logger.error("No labels available for retraining")
            return current_model
        
        # Get retraining policy (from last decision)
        if not self.decision_engine.decision_history:
            logger.error("No retraining decision available")
            return current_model
        
        last_decision = self.decision_engine.decision_history[-1]
        
        # Create policy object
        from decision_engine import RetrainingPolicy
        policy = RetrainingPolicy(
            decision=RetrainingDecision(last_decision['decision']),
            confidence=last_decision['confidence'],
            data_window_start=last_decision['window'][0],
            data_window_end=last_decision['window'][1],
            components_to_update=['all'],
            reasoning=last_decision['reasoning'],
            estimated_cost=1.0
        )
        
        # Execute retraining
        new_model = self.retraining_orchestrator.execute_retraining(
            current_model=current_model,
            new_data=new_data,
            new_labels=new_labels,
            retraining_policy=policy,
            train_fn=train_fn,
            get_gradients_fn=get_gradients_fn,
            get_params_fn=get_params_fn,
            old_data=self.reference_data,
            old_labels=self.reference_labels
        )
        
        # Validate new model
        logger.info("Validating new model...")
        safety_report = self.validator.validate(
            champion_model=current_model,
            challenger_model=new_model,
            validation_data=validation_data,
            validation_labels=validation_labels,
            metrics_fn=metrics_fn,
            predict_fn=predict_fn
        )
        
        logger.info(f"Validation Result: {safety_report.validation_result.value}")
        logger.info(f"Recommendation: {safety_report.recommendation}")
        
        # Make deployment decision
        if safety_report.validation_result == ValidationResult.PASS:
            logger.info("✓ New model ACCEPTED - deploying")
            
            # Register new model
            new_metrics = safety_report.challenger_performance
            version_id = self.retraining_orchestrator.register_trained_model(
                new_model, new_metrics, {}, new_data
            )
            
            self.state.current_model_version = version_id
            self.state.performance_history.append({
                'timestamp': datetime.now(),
                'version': version_id,
                'metrics': new_metrics
            })
            
            # Provide feedback to RL policy
            if self.config.enable_rl_policy:
                improvement = (new_metrics.get('accuracy', 0) - 
                             safety_report.champion_performance.get('accuracy', 0))
                self.decision_engine.provide_feedback(improvement, policy.estimated_cost)
            
            return new_model
        
        elif safety_report.validation_result == ValidationResult.FAIL:
            logger.warning("✗ New model REJECTED - rolling back")
            
            if self.rollback_manager.can_rollback():
                rollback_model = self.rollback_manager.rollback(
                    safety_report.recommendation,
                    "challenger_failed"
                )
                self.state.rollback_count += 1
                
                # Negative feedback to RL policy
                if self.config.enable_rl_policy:
                    self.decision_engine.provide_feedback(-0.1, policy.estimated_cost)
                
                return rollback_model
            else:
                logger.error("Cannot rollback - no previous version available")
                return current_model
        
        else:  # INCONCLUSIVE
            logger.info("⚠ Validation inconclusive - keeping current model, monitoring")
            return current_model
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            'current_version': self.state.current_model_version,
            'last_retrain': self.state.last_retrain_timestamp.isoformat() if self.state.last_retrain_timestamp else None,
            'drift_detected_count': self.state.drift_detected_count,
            'retraining_count': self.state.retraining_count,
            'rollback_count': self.state.rollback_count,
            'buffer_size': len(self.current_data_buffer),
            'performance_history': self.state.performance_history[-5:]  # Last 5 entries
        }
    
    def export_metrics(self) -> Dict:
        """Export metrics for monitoring"""
        return {
            'drift_detection_rate': self.state.drift_detected_count / max(self.state.retraining_count, 1),
            'retraining_frequency': self.state.retraining_count,
            'rollback_rate': self.state.rollback_count / max(self.state.retraining_count, 1),
            'current_buffer_utilization': len(self.current_data_buffer) / self.config.drift_detection_window
        }
