"""
Drift-Aware Decision Engine
Intelligent retraining policy with rule-based and RL-based strategies
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class RetrainingDecision(Enum):
    NO_RETRAIN = "no_retrain"
    PARTIAL_RETRAIN = "partial_retrain"
    FULL_RETRAIN = "full_retrain"
    INCREMENTAL_UPDATE = "incremental_update"

@dataclass
class RetrainingPolicy:
    decision: RetrainingDecision
    confidence: float
    data_window_start: int
    data_window_end: int
    components_to_update: List[str]
    reasoning: str
    estimated_cost: float

class RuleBasedPolicy:
    """Rule-based retraining policy - baseline and fallback"""
    def __init__(self, 
                 severity_threshold_full: float = 0.5,
                 severity_threshold_partial: float = 0.3,
                 min_samples: int = 1000,
                 max_retraining_frequency_days: int = 7):
        self.severity_threshold_full = severity_threshold_full
        self.severity_threshold_partial = severity_threshold_partial
        self.min_samples = min_samples
        self.max_retraining_frequency_days = max_retraining_frequency_days
        self.last_retrain_timestamp = None
    
    def decide(self, 
               severity: float,
               confidence: float,
               available_samples: int,
               days_since_last_retrain: Optional[int] = None,
               feature_attributions: Optional[Dict[str, float]] = None) -> RetrainingPolicy:
        """Rule-based decision logic"""
        
        # Safety checks
        if available_samples < self.min_samples:
            return RetrainingPolicy(
                decision=RetrainingDecision.NO_RETRAIN,
                confidence=1.0,
                data_window_start=0,
                data_window_end=available_samples,
                components_to_update=[],
                reasoning=f"Insufficient samples ({available_samples} < {self.min_samples})",
                estimated_cost=0.0
            )
        
        if days_since_last_retrain and days_since_last_retrain < self.max_retraining_frequency_days:
            return RetrainingPolicy(
                decision=RetrainingDecision.NO_RETRAIN,
                confidence=1.0,
                data_window_start=0,
                data_window_end=available_samples,
                components_to_update=[],
                reasoning=f"Too soon since last retrain ({days_since_last_retrain} < {self.max_retraining_frequency_days} days)",
                estimated_cost=0.0
            )
        
        # Decision logic
        if severity >= self.severity_threshold_full and confidence > 0.7:
            decision = RetrainingDecision.FULL_RETRAIN
            window_start = max(0, available_samples - 10000)
            components = ['all']
            reasoning = f"High severity ({severity:.3f}) with high confidence ({confidence:.3f})"
            cost = 1.0
        
        elif severity >= self.severity_threshold_partial and confidence > 0.6:
            decision = RetrainingDecision.PARTIAL_RETRAIN
            window_start = max(0, available_samples - 5000)
            components = self._identify_components_to_update(feature_attributions)
            reasoning = f"Medium severity ({severity:.3f}) - partial retrain on {len(components)} components"
            cost = 0.5
        
        elif severity >= 0.15 and confidence > 0.5:
            decision = RetrainingDecision.INCREMENTAL_UPDATE
            window_start = max(0, available_samples - 2000)
            components = ['output_layer']
            reasoning = f"Low severity ({severity:.3f}) - incremental update only"
            cost = 0.2
        
        else:
            decision = RetrainingDecision.NO_RETRAIN
            window_start = 0
            components = []
            reasoning = f"Severity below threshold ({severity:.3f} < {self.severity_threshold_partial})"
            cost = 0.0
        
        return RetrainingPolicy(
            decision=decision,
            confidence=confidence,
            data_window_start=window_start,
            data_window_end=available_samples,
            components_to_update=components,
            reasoning=reasoning,
            estimated_cost=cost
        )
    
    def _identify_components_to_update(self, feature_attributions: Optional[Dict[str, float]]) -> List[str]:
        """Identify which model components need updating based on drift attribution"""
        if not feature_attributions:
            return ['all']
        
        components = []
        top_features = sorted(feature_attributions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for feature, attribution in top_features:
            if attribution > 0.3:
                components.append(f"feature_{feature}")
        
        if not components:
            components = ['output_layer']
        
        return components

class RLBasedPolicy:
    """Reinforcement Learning-based retraining policy (using simple Q-learning)"""
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, epsilon: float = 0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # State -> Action -> Q-value
        self.state_history = []
        self.action_history = []
        self.reward_history = []
    
    def _discretize_state(self, severity: float, confidence: float, days_since_retrain: int) -> str:
        """Convert continuous state to discrete state key"""
        sev_bucket = int(severity * 10)
        conf_bucket = int(confidence * 10)
        days_bucket = min(days_since_retrain // 7, 4)  # 0-4 weeks
        return f"{sev_bucket}_{conf_bucket}_{days_bucket}"
    
    def _get_q_value(self, state: str, action: RetrainingDecision) -> float:
        """Get Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in RetrainingDecision}
        return self.q_table[state][action]
    
    def _set_q_value(self, state: str, action: RetrainingDecision, value: float):
        """Set Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in RetrainingDecision}
        self.q_table[state][action] = value
    
    def decide(self, severity: float, confidence: float, days_since_retrain: int = 0) -> RetrainingDecision:
        """RL-based decision with epsilon-greedy exploration"""
        state = self._discretize_state(severity, confidence, days_since_retrain)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.choice(list(RetrainingDecision))
        else:
            q_values = {a: self._get_q_value(state, a) for a in RetrainingDecision}
            action = max(q_values, key=q_values.get)
        
        self.state_history.append(state)
        self.action_history.append(action)
        
        return action
    
    def update(self, reward: float):
        """Update Q-values based on observed reward"""
        if len(self.state_history) < 2:
            return
        
        state = self.state_history[-2]
        action = self.action_history[-2]
        next_state = self.state_history[-1]
        
        # Q-learning update
        current_q = self._get_q_value(state, action)
        max_next_q = max(self._get_q_value(next_state, a) for a in RetrainingDecision)
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        
        self._set_q_value(state, action, new_q)
        self.reward_history.append(reward)
    
    def save_policy(self, filepath: str):
        """Save learned Q-table"""
        with open(filepath, 'w') as f:
            serializable_q_table = {
                state: {action.value: q_val for action, q_val in actions.items()}
                for state, actions in self.q_table.items()
            }
            json.dump(serializable_q_table, f)
    
    def load_policy(self, filepath: str):
        """Load learned Q-table"""
        with open(filepath, 'r') as f:
            serializable_q_table = json.load(f)
            self.q_table = {
                state: {RetrainingDecision(action): q_val for action, q_val in actions.items()}
                for state, actions in serializable_q_table.items()
            }

class AdaptiveWindowSelector:
    """Selects optimal data window for retraining"""
    def __init__(self, min_window: int = 1000, max_window: int = 50000):
        self.min_window = min_window
        self.max_window = max_window
    
    def select_window(self, 
                     total_samples: int,
                     drift_severity: float,
                     temporal_segments: List[Dict]) -> Tuple[int, int]:
        """Select data window based on drift characteristics"""
        
        if not temporal_segments:
            # No temporal localization - use recent data
            window_size = int(self.min_window + (self.max_window - self.min_window) * drift_severity)
            window_size = min(window_size, total_samples)
            return max(0, total_samples - window_size), total_samples
        
        # Use temporal segments to define window
        earliest_drift = min(seg['start_idx'] for seg in temporal_segments)
        
        # Start from earliest drift point
        start_idx = max(0, earliest_drift - self.min_window // 2)
        end_idx = total_samples
        
        return start_idx, end_idx

class DecisionEngine:
    """Main decision engine orchestrator"""
    def __init__(self, use_rl: bool = False):
        self.rule_based_policy = RuleBasedPolicy()
        self.rl_policy = RLBasedPolicy() if use_rl else None
        self.window_selector = AdaptiveWindowSelector()
        self.use_rl = use_rl
        self.decision_history = []
    
    def decide(self,
               severity_analysis: any,
               available_samples: int,
               days_since_last_retrain: Optional[int] = None,
               performance_metrics: Optional[Dict] = None) -> RetrainingPolicy:
        """Make intelligent retraining decision"""
        
        severity = severity_analysis.overall_severity
        confidence = (severity_analysis.confidence_interval[0] + severity_analysis.confidence_interval[1]) / 2
        feature_attributions = severity_analysis.feature_attributions
        temporal_segments = severity_analysis.temporal_segments
        
        # Rule-based decision (always computed as fallback)
        rule_policy = self.rule_based_policy.decide(
            severity=severity,
            confidence=confidence,
            available_samples=available_samples,
            days_since_last_retrain=days_since_last_retrain,
            feature_attributions=feature_attributions
        )
        
        # RL-based decision (if enabled and trained)
        if self.use_rl and self.rl_policy and len(self.rl_policy.q_table) > 10:
            rl_decision = self.rl_policy.decide(severity, confidence, days_since_last_retrain or 0)
            
            # Blend RL and rule-based decisions
            if rl_decision != rule_policy.decision:
                # Use RL decision but with rule-based parameters
                rule_policy.decision = rl_decision
                rule_policy.reasoning += f" | RL override: {rl_decision.value}"
        
        # Adaptive window selection
        window_start, window_end = self.window_selector.select_window(
            available_samples, severity, temporal_segments
        )
        rule_policy.data_window_start = window_start
        rule_policy.data_window_end = window_end
        
        # Log decision
        self.decision_history.append({
            'severity': severity,
            'confidence': confidence,
            'decision': rule_policy.decision.value,
            'window': (window_start, window_end),
            'reasoning': rule_policy.reasoning
        })
        
        return rule_policy
    
    def provide_feedback(self, performance_improvement: float, retraining_cost: float):
        """Provide feedback to RL policy"""
        if self.use_rl and self.rl_policy:
            # Reward = performance improvement - cost penalty
            reward = performance_improvement - 0.1 * retraining_cost
            self.rl_policy.update(reward)
