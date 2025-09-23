# Metrics Collection Implementation Plan

## Overview

This document outlines the implementation plan for adding comprehensive metrics collection to the ClStock system. The goal is to provide visibility into system performance, business metrics, and operational health.

## Metrics Categories

### 1. Business Metrics
- **Prediction Accuracy**: Track accuracy of stock predictions
- **API Usage**: Monitor API endpoint usage and response times
- **User Engagement**: Track user interactions with the system
- **Recommendation Quality**: Measure the quality of generated recommendations

### 2. System Performance Metrics
- **Response Times**: API response time percentiles
- **Throughput**: Requests per second
- **Error Rates**: HTTP error rates by endpoint
- **Resource Usage**: CPU, memory, disk, and network utilization

### 3. Model Performance Metrics
- **Model Training Time**: Time required to train models
- **Prediction Latency**: Time to generate predictions
- **Feature Importance**: Track changes in feature importance
- **Model Drift**: Monitor for changes in data distribution

## Implementation Approach

### Phase 1: Infrastructure Setup

#### 1. Metrics Collection Library
Implement a metrics collection abstraction that can support multiple backends:

```python
# utils/metrics.py
import time
from typing import Dict, Any
from abc import ABC, abstractmethod

class MetricsCollector(ABC):
    @abstractmethod
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        pass
    
    @abstractmethod
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        pass
    
    @abstractmethod
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        pass

class PrometheusMetricsCollector(MetricsCollector):
    # Implementation for Prometheus
    pass

class StatsDMetricsCollector(MetricsCollector):
    # Implementation for StatsD
    pass
```

#### 2. Global Metrics Instance
Create a global metrics instance that can be used throughout the application:

```python
# utils/metrics.py
from typing import Optional

_metrics_collector: Optional[MetricsCollector] = None

def set_metrics_collector(collector: MetricsCollector):
    global _metrics_collector
    _metrics_collector = collector

def get_metrics_collector() -> Optional[MetricsCollector]:
    return _metrics_collector
```

### Phase 2: Business Metrics Implementation

#### 1. Prediction Accuracy Metrics
Track prediction accuracy by symbol and time period:

```python
# models/predictor.py
from utils.metrics import get_metrics_collector

class StockPredictor:
    def calculate_score(self, symbol: str) -> float:
        # Existing implementation
        score = self._calculate_score_impl(symbol)
        
        # Record metrics
        metrics = get_metrics_collector()
        if metrics:
            metrics.increment_counter("prediction.calculated", 1, {"symbol": symbol})
            metrics.record_gauge("prediction.score", score, {"symbol": symbol})
        
        return score
```

#### 2. API Metrics
Add metrics to API endpoints:

```python
# api/endpoints.py
from utils.metrics import get_metrics_collector
import time

@router.get("/recommendations")
async def get_recommendations(top_n: int = Query(5, ge=1, le=10)):
    metrics = get_metrics_collector()
    start_time = time.time()
    
    try:
        # Existing implementation
        predictor = StockPredictor()
        recommendations = predictor.get_top_recommendations(top_n)
        
        # Record success metrics
        if metrics:
            duration = time.time() - start_time
            metrics.record_timer("api.recommendations.response_time", duration)
            metrics.increment_counter("api.recommendations.success", 1)
            metrics.record_gauge("api.recommendations.count", len(recommendations))
        
        return recommendations
    except Exception as e:
        # Record error metrics
        if metrics:
            duration = time.time() - start_time
            metrics.record_timer("api.recommendations.response_time", duration)
            metrics.increment_counter("api.recommendations.error", 1, {"error_type": type(e).__name__})
        raise
```

### Phase 3: System Performance Metrics

#### 1. Resource Usage Monitoring
Extend the existing system monitor to export metrics:

```python
# monitoring/system_monitor.py
from utils.metrics import get_metrics_collector

class SystemMonitor:
    def _collect_system_metrics(self) -> SystemMetrics:
        # Existing implementation
        metrics = self._collect_system_metrics_impl()
        
        # Export to metrics collector
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_gauge("system.cpu.percent", metrics.cpu_percent)
            metrics_collector.record_gauge("system.memory.percent", metrics.memory_percent)
            metrics_collector.record_gauge("system.disk.percent", metrics.disk_usage_percent)
        
        return metrics
```

#### 2. Model Performance Metrics
Add metrics to ML model training and prediction:

```python
# models/ml_models.py
from utils.metrics import get_metrics_collector
import time

class MLStockPredictor:
    def train(self, symbols: List[str]):
        metrics = get_metrics_collector()
        start_time = time.time()
        
        try:
            # Existing training implementation
            self._train_impl(symbols)
            
            # Record training metrics
            if metrics:
                duration = time.time() - start_time
                metrics.record_timer("model.training.duration", duration)
                metrics.increment_counter("model.training.success", 1)
                
                # Record model performance metrics
                if hasattr(self.model, "score"):
                    score = self.model.score(X_test, y_test)
                    metrics.record_gauge("model.training.score", score)
        except Exception as e:
            if metrics:
                duration = time.time() - start_time
                metrics.record_timer("model.training.duration", duration)
                metrics.increment_counter("model.training.error", 1, {"error_type": type(e).__name__})
            raise
```

## Metrics Export Configuration

### Prometheus Export
Set up Prometheus metrics export endpoint:

```python
# app/main.py
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### StatsD Export
Configure StatsD client for metrics export:

```python
# config/settings.py
@dataclass
class MetricsConfig:
    backend: str = "prometheus"  # or "statsd"
    prometheus_port: int = 8001
    statsd_host: str = "localhost"
    statsd_port: int = 8125
    prefix: str = "clstock"
```

## Dashboard Creation

### Grafana Dashboards
Create Grafana dashboards for key metrics:

1. **Business Dashboard**
   - Prediction accuracy trends
   - API usage statistics
   - Recommendation quality metrics

2. **System Performance Dashboard**
   - CPU and memory usage
   - API response times
   - Error rates and trends

3. **Model Performance Dashboard**
   - Training times and success rates
   - Model accuracy metrics
   - Feature importance tracking

## Alerting Rules

### Critical Alerts
- API error rate > 5%
- System CPU usage > 90% for 5 minutes
- Memory usage > 95% for 5 minutes
- Prediction accuracy < 70% for 1 hour

### Warning Alerts
- API response time > 2 seconds
- System CPU usage > 80% for 10 minutes
- Memory usage > 85% for 10 minutes
- Model training failures

## Implementation Timeline

### Week 1
- Set up metrics collection infrastructure
- Implement basic metrics collector abstraction
- Add global metrics instance

### Week 2
- Implement business metrics collection
- Add prediction accuracy tracking
- Add API metrics collection

### Week 3
- Implement system performance metrics
- Extend system monitor with metrics export
- Add model performance metrics

### Week 4
- Configure metrics export (Prometheus/StatsD)
- Create Grafana dashboards
- Set up alerting rules

### Week 5
- Performance testing and optimization
- Documentation and training
- Production deployment

## Success Metrics

1. **Coverage**: >90% of critical business and system metrics collected
2. **Performance**: <5% overhead on API response times
3. **Reliability**: >99.9% metrics collection uptime
4. **Actionability**: Reduction in mean time to detect issues by 50%