"""
Complete Example: Drift-Aware Retraining Pipeline
Demonstrates full system with synthetic drift scenarios
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')

from main_pipeline import DriftAwareRetrainingPipeline, PipelineConfig

def create_synthetic_data_with_drift(n_samples=5000, drift_type='sudden', drift_point=2500):
    """Create synthetic dataset with controlled drift"""
    # Initial distribution
    X1, y1 = make_classification(
        n_samples=drift_point,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
        flip_y=0.1
    )
    
    if drift_type == 'sudden':
        # Sudden drift: completely different distribution
        X2, y2 = make_classification(
            n_samples=n_samples - drift_point,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=123,  # Different seed
            flip_y=0.2  # More noise
        )
    elif drift_type == 'gradual':
        # Gradual drift: slowly changing distribution
        X2, y2 = make_classification(
            n_samples=n_samples - drift_point,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42,
            flip_y=0.15
        )
        # Add gradual shift
        shift = np.linspace(0, 2, n_samples - drift_point).reshape(-1, 1)
        X2 = X2 + shift
    else:  # no_drift
        X2, y2 = make_classification(
            n_samples=n_samples - drift_point,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42,
            flip_y=0.1
        )
    
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    
    return X, y, drift_point

def train_model(model, X, y, policy=None, incremental_trainer=None):
    """Training function"""
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    return model

def predict_model(model, X):
    """Prediction function"""
    return model.predict(X)

def compute_metrics(predictions, labels):
    """Compute evaluation metrics"""
    if labels is None:
        return {}
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

def get_gradients(model, X, y):
    """Placeholder for gradient computation"""
    # Random Forest doesn't have gradients, return dummy
    return {'dummy': np.zeros(10)}

def get_params(model):
    """Get model parameters"""
    return {'dummy': np.zeros(10)}

def run_example():
    """Run complete pipeline example"""
    print("=" * 80)
    print("DRIFT-AWARE RETRAINING PIPELINE - COMPLETE EXAMPLE")
    print("=" * 80)
    
    # Configuration
    config = PipelineConfig(
        drift_detection_window=500,
        min_samples_for_retrain=500,
        enable_rl_policy=False,
        max_retraining_frequency_days=0  # Allow immediate retraining for demo
    )
    
    # Initialize pipeline
    pipeline = DriftAwareRetrainingPipeline(config)
    
    # Create synthetic data with sudden drift
    print("\n[1] Generating synthetic data with SUDDEN DRIFT...")
    X, y, drift_point = create_synthetic_data_with_drift(
        n_samples=5000,
        drift_type='sudden',
        drift_point=2500
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X[:drift_point], y[:drift_point], test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Post-drift samples: {len(X) - drift_point}")
    
    # Train initial model
    print("\n[2] Training initial model...")
    initial_model = train_model(None, X_train, y_train)
    
    # Evaluate initial model
    train_preds = predict_model(initial_model, X_train)
    train_metrics = compute_metrics(train_preds, y_train)
    print(f"   Initial model performance: Accuracy={train_metrics['accuracy']:.4f}, F1={train_metrics['f1_score']:.4f}")
    
    # Set reference data
    print("\n[3] Setting reference data for drift detection...")
    pipeline.set_reference_data(X_train, y_train)
    pipeline.reference_predictions = predict_model(initial_model, X_train)
    
    # Simulate production: monitor predictions on post-drift data
    print("\n[4] Simulating production monitoring (post-drift data)...")
    post_drift_X = X[drift_point:]
    post_drift_y = y[drift_point:]
    
    current_model = initial_model
    
    # Monitor in batches
    batch_size = 100
    for i in range(0, len(post_drift_X), batch_size):
        batch_X = post_drift_X[i:i+batch_size]
        batch_y = post_drift_y[i:i+batch_size]
        
        # Make predictions and monitor
        for j in range(len(batch_X)):
            pred = predict_model(current_model, batch_X[j:j+1])
            pipeline.monitor_prediction(
                batch_X[j],
                pred,
                batch_y[j] if j < len(batch_y) else None
            )
        
        # Check if retraining was triggered
        if pipeline.state.drift_detected_count > 0 and pipeline.state.retraining_count == 0:
            print(f"\n   [!] DRIFT DETECTED at sample {drift_point + i}")
            print(f"   Drift detection count: {pipeline.state.drift_detected_count}")
            
            # Execute full retraining cycle
            print("\n[5] Executing retraining cycle...")
            
            # Prepare validation data
            val_X = X_test
            val_y = y_test
            
            # Execute retraining
            new_model = pipeline.execute_full_retraining_cycle(
                current_model=current_model,
                train_fn=train_model,
                predict_fn=predict_model,
                metrics_fn=compute_metrics,
                get_gradients_fn=get_gradients,
                get_params_fn=get_params,
                validation_data=val_X,
                validation_labels=val_y
            )
            
            if new_model != current_model:
                print("   [OK] Model updated successfully")
                current_model = new_model
            else:
                print("   [!] Model not updated (validation failed or inconclusive)")
            
            break
    
    # Final evaluation
    print("\n[6] Final Evaluation...")
    
    # Evaluate on post-drift data
    post_drift_preds = predict_model(current_model, post_drift_X[:1000])
    post_drift_metrics = compute_metrics(post_drift_preds, post_drift_y[:1000])
    
    print(f"   Post-drift performance: Accuracy={post_drift_metrics['accuracy']:.4f}, F1={post_drift_metrics['f1_score']:.4f}")
    
    # Pipeline status
    print("\n[7] Pipeline Status:")
    status = pipeline.get_pipeline_status()
    for key, value in status.items():
        if key != 'performance_history':
            print(f"   {key}: {value}")
    
    # Metrics
    print("\n[8] Pipeline Metrics:")
    metrics = pipeline.export_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_example()
