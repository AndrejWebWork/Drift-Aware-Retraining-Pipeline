"""
Drift-Aware Retraining Pipeline
World-Class, Production-Ready, 100% Free MLOps System

Zero Cost | Research-Grade | Production-Ready | Fully Auditable
"""

__version__ = "1.0.0"
__author__ = "Open Source Contributors"
__license__ = "Apache-2.0"

# Main pipeline
from .main_pipeline import (
    DriftAwareRetrainingPipeline,
    PipelineConfig,
    PipelineState
)

# Drift detection
from .drift_detection import (
    DriftEnsemble,
    KSDetector,
    PSIDetector,
    JSDetector,
    MMDDetector,
    ADWINDetector,
    PredictionDriftDetector,
    DriftType,
    DriftSeverity,
    DriftResult
)

# Severity analysis
from .severity_analysis import (
    SeverityAnalyzer,
    BayesianSeverityScorer,
    FeatureAttributor,
    TemporalLocalizer,
    RootCauseAnalyzer,
    SeverityAnalysis
)

# Decision engine
from .decision_engine import (
    DecisionEngine,
    RuleBasedPolicy,
    RLBasedPolicy,
    AdaptiveWindowSelector,
    RetrainingDecision,
    RetrainingPolicy
)

# Retraining pipeline
from .retraining_pipeline import (
    RetrainingOrchestrator,
    IncrementalTrainer,
    ElasticWeightConsolidation,
    LearningWithoutForgetting,
    DataCurator,
    ModelRegistry,
    ModelVersion
)

# Safety validation
from .safety_validation import (
    ChampionChallengerValidator,
    ShadowEvaluator,
    StatisticalSignificanceTester,
    RegressionDetector,
    AutomaticRollback,
    ValidationResult,
    SafetyReport
)

__all__ = [
    # Main
    'DriftAwareRetrainingPipeline',
    'PipelineConfig',
    'PipelineState',
    
    # Drift detection
    'DriftEnsemble',
    'KSDetector',
    'PSIDetector',
    'JSDetector',
    'MMDDetector',
    'ADWINDetector',
    'PredictionDriftDetector',
    'DriftType',
    'DriftSeverity',
    'DriftResult',
    
    # Severity
    'SeverityAnalyzer',
    'BayesianSeverityScorer',
    'FeatureAttributor',
    'TemporalLocalizer',
    'RootCauseAnalyzer',
    'SeverityAnalysis',
    
    # Decision
    'DecisionEngine',
    'RuleBasedPolicy',
    'RLBasedPolicy',
    'AdaptiveWindowSelector',
    'RetrainingDecision',
    'RetrainingPolicy',
    
    # Retraining
    'RetrainingOrchestrator',
    'IncrementalTrainer',
    'ElasticWeightConsolidation',
    'LearningWithoutForgetting',
    'DataCurator',
    'ModelRegistry',
    'ModelVersion',
    
    # Safety
    'ChampionChallengerValidator',
    'ShadowEvaluator',
    'StatisticalSignificanceTester',
    'RegressionDetector',
    'AutomaticRollback',
    'ValidationResult',
    'SafetyReport',
]

# Package metadata
PACKAGE_INFO = {
    'name': 'drift-aware-retraining-pipeline',
    'version': __version__,
    'description': 'World-class drift-aware retraining pipeline - 100% free',
    'author': __author__,
    'license': __license__,
    'url': 'https://github.com/...',
    'keywords': [
        'mlops', 'drift-detection', 'model-retraining', 
        'continual-learning', 'machine-learning', 'production-ml'
    ],
    'classifiers': [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
}

def get_version():
    """Get package version"""
    return __version__

def get_info():
    """Get package information"""
    return PACKAGE_INFO
