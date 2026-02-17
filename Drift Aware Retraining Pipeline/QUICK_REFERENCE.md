# ⚡ QUICK REFERENCE GUIDE
## Drift-Aware Retraining Pipeline - Cheat Sheet

---

## 🚀 30-SECOND START

```bash
pip install -r requirements.txt && python example_usage.py
```

---

## 📝 MINIMAL INTEGRATION (5 lines)

```python
from main_pipeline import DriftAwareRetrainingPipeline, PipelineConfig

pipeline = DriftAwareRetrainingPipeline(PipelineConfig())
pipeline.set_reference_data(X_train, y_train)

# In production loop:
pipeline.monitor_prediction(features, prediction, true_label)
```

---

## 🎛️ CONFIGURATION PRESETS

### Conservative (Low False Positives)
```python
config = PipelineConfig(
    drift_detection_window=2000,
    max_retraining_frequency_days=14
)
```

### Aggressive (Fast Response)
```python
config = PipelineConfig(
    drift_detection_window=500,
    max_retraining_frequency_days=3
)
```

### Production (Balanced)
```python
config = PipelineConfig(
    drift_detection_window=1000,
    max_retraining_frequency_days=7,
    enable_rl_policy=True
)
```

---

## 🔍 DRIFT DETECTION THRESHOLDS

| Detector | No Drift | Low | Medium | High | Critical |
|----------|----------|-----|--------|------|----------|
| KS-Test | < 0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 | > 0.5 |
| PSI | < 0.1 | 0.1-0.2 | 0.2-0.25 | > 0.25 | > 0.3 |
| JS-Div | < 0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 | > 0.5 |
| MMD | < 0.05 | 0.05-0.1 | 0.1-0.2 | 0.2-0.4 | > 0.4 |

---

## 🎯 RETRAINING DECISION RULES

```
IF severity >= 0.5 AND confidence > 0.7:
    → FULL_RETRAIN (all components, large window)

ELIF severity >= 0.3 AND confidence > 0.6:
    → PARTIAL_RETRAIN (top features, medium window)

ELIF severity >= 0.15 AND confidence > 0.5:
    → INCREMENTAL_UPDATE (output layer, small window)

ELSE:
    → NO_RETRAIN (continue monitoring)
```

---

## 🛡️ SAFETY VALIDATION CRITERIA

**Deploy New Model IF:**
1. ✅ Statistical significance (p < 0.05)
2. ✅ Performance improvement on primary metric
3. ✅ No regression > 2% on critical metrics
4. ✅ Minimum 1000 validation samples

**Rollback IF:**
1. ❌ Performance regression detected
2. ❌ Statistical test fails
3. ❌ Validation inconclusive after 3 attempts

---

## 📊 MONITORING METRICS

### Key Metrics to Track
```python
metrics = pipeline.export_metrics()

# Critical:
- drift_detection_rate: Should be 5-15%
- rollback_rate: Should be < 5%
- retraining_frequency: Depends on domain

# Operational:
- buffer_utilization: Should be 80-100%
- validation_pass_rate: Should be > 90%
```

### Prometheus Queries
```promql
# Drift detection rate
rate(drift_detected_total[1h])

# Retraining frequency
rate(retraining_triggered_total[24h])

# Model performance
model_accuracy_score
```

---

## 🐛 TROUBLESHOOTING

### Problem: Too Many False Positives
```python
# Solution 1: Increase thresholds
config.drift_detection_window = 2000  # Larger window

# Solution 2: Adjust detector sensitivity
from drift_detection import KSDetector
detector = KSDetector(alpha=0.01)  # More conservative
```

### Problem: Missing Real Drift
```python
# Solution 1: Decrease window size
config.drift_detection_window = 500  # More sensitive

# Solution 2: Enable more detectors
ensemble.detectors['adwin'] = ADWINDetector(delta=0.001)
```

### Problem: Retraining Too Frequent
```python
# Solution: Increase frequency limit
config.max_retraining_frequency_days = 14  # Max once per 2 weeks
```

### Problem: Retraining Not Triggered
```python
# Solution: Lower severity threshold
policy = RuleBasedPolicy(
    severity_threshold_full=0.4,  # Lower from 0.5
    severity_threshold_partial=0.2  # Lower from 0.3
)
```

---

## 🔧 COMMON CUSTOMIZATIONS

### Custom Drift Detector
```python
class CustomDetector:
    def detect(self, reference, current, feature_name=None):
        # Your logic here
        return DriftResult(...)

ensemble.detectors['custom'] = CustomDetector()
```

### Custom Retraining Policy
```python
class CustomPolicy:
    def decide(self, severity, confidence, available_samples):
        # Your logic here
        return RetrainingPolicy(...)

engine.rule_based_policy = CustomPolicy()
```

### Custom Metrics
```python
def custom_metrics(predictions, labels):
    return {
        'accuracy': accuracy_score(labels, predictions),
        'custom_metric': your_metric(labels, predictions)
    }
```

---

## 📦 DEPLOYMENT CHECKLIST

### Pre-Deployment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run tests: `pytest test_pipeline.py`
- [ ] Configure pipeline: Set appropriate thresholds
- [ ] Set reference data: `pipeline.set_reference_data()`
- [ ] Test on historical data

### Deployment
- [ ] Deploy container: `docker run drift-pipeline`
- [ ] Configure monitoring: Prometheus + Grafana
- [ ] Set up alerts: Drift detection, rollback events
- [ ] Enable logging: Audit trail
- [ ] Test rollback: Verify automatic rollback works

### Post-Deployment
- [ ] Monitor drift detection rate
- [ ] Track retraining frequency
- [ ] Review validation results
- [ ] Analyze performance trends
- [ ] Tune thresholds as needed

---

## 🎓 BEST PRACTICES

### DO ✅
- Use ensemble of detectors (not single)
- Set reference data from stable period
- Monitor multiple metrics
- Enable automatic rollback
- Log all decisions
- Test on synthetic drift first
- Validate before deployment

### DON'T ❌
- Rely on single detector
- Set reference data during drift
- Ignore validation failures
- Disable safety checks
- Skip testing
- Deploy without monitoring
- Ignore false positives

---

## 📞 EMERGENCY CONTACTS

### System Down
```bash
# Check logs
tail -f logs/pipeline.log

# Restart pipeline
docker restart drift-pipeline

# Rollback to last known good
pipeline.rollback_manager.rollback("emergency", "current")
```

### Performance Degradation
```bash
# Check current status
status = pipeline.get_pipeline_status()
print(status)

# Force retraining
pipeline.execute_full_retraining_cycle(...)

# Adjust thresholds
config.drift_detection_window = 2000
```

### Data Quality Issues
```bash
# Check data curator settings
curator = DataCurator(
    min_confidence=0.7,  # Increase quality threshold
    max_samples=10000
)

# Enable class balancing
curated_data, curated_labels = curator.balance_classes(data, labels)
```

---

## 🔗 QUICK LINKS

- **Full Documentation:** README.md
- **Example Code:** example_usage.py
- **Test Suite:** test_pipeline.py
- **Architecture:** WORLD_CLASS_SUMMARY.md
- **Dependencies:** requirements.txt
- **Docker:** Dockerfile

---

## 💡 PRO TIPS

1. **Start Conservative:** Use larger windows and higher thresholds initially
2. **Monitor First Week:** Tune based on false positive/negative rates
3. **Enable RL After 100 Retrainings:** Let it learn your patterns
4. **Use Shadow Mode:** Test new models before deployment
5. **Keep Reference Data Fresh:** Update every 3-6 months
6. **Log Everything:** You'll thank yourself during debugging
7. **Test Rollback:** Verify it works before you need it
8. **Scale Horizontally:** Multiple instances > single large instance

---

## 📈 PERFORMANCE TUNING

### For High Throughput
```python
config = PipelineConfig(
    drift_detection_window=5000,  # Larger batches
    validation_split=0.1  # Faster validation
)
```

### For Low Latency
```python
# Use fewer detectors
ensemble.detectors = {'ks': KSDetector(), 'psi': PSIDetector()}
```

### For High Accuracy
```python
# Use all detectors + RL
config = PipelineConfig(
    enable_rl_policy=True,
    drift_detection_window=1000
)
```

---

## 🎯 SUCCESS METRICS

**Week 1:** System deployed, monitoring active
**Week 2:** First drift detected and handled
**Month 1:** Retraining policy tuned, < 5% false positives
**Month 3:** RL policy trained, automatic optimization
**Month 6:** Zero manual interventions, full automation

---

**⚡ REMEMBER: Start simple, monitor closely, tune gradually, automate completely.**

**🌍 You now have world-class MLOps. Use it wisely.**
