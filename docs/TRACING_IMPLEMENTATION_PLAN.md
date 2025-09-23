# Distributed Tracing Implementation Plan

## Overview

This document outlines the implementation plan for adding distributed tracing to the ClStock system using OpenTelemetry. The goal is to provide end-to-end visibility into complex workflows and enable performance debugging across service boundaries.

## Tracing Requirements

### Use Cases to Trace
1. **API Request Flow**: Trace from API entry point through to response
2. **Prediction Workflows**: Trace the complete prediction process for a stock symbol
3. **Data Fetching**: Trace data retrieval from external sources
4. **Model Training**: Trace machine learning model training processes
5. **Batch Processing**: Trace batch prediction workflows

### Key Information to Capture
- Request/response flow through components
- Timing information for each operation
- Error propagation and handling
- Context propagation across async operations
- Correlation between related operations

## Implementation Approach

### Phase 1: OpenTelemetry Setup

#### 1. Dependency Installation
Add OpenTelemetry dependencies to requirements.txt:

```txt
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-jaeger>=1.20.0
opentelemetry-exporter-zipkin>=1.20.0
opentelemetry-instrumentation-fastapi>=0.41b0
opentelemetry-instrumentation-requests>=0.41b0
```

#### 2. Tracer Configuration
Create tracer configuration module:

```python
# utils/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import logging

logger = logging.getLogger(__name__)

def setup_tracing(service_name: str = "clstock"):
    """Set up OpenTelemetry tracing."""
    try:
        # Create tracer provider
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(span_processor)
        
        logger.info("Tracing setup completed")
    except Exception as e:
        logger.error(f"Failed to set up tracing: {e}")

def get_tracer():
    """Get tracer instance."""
    return trace.get_tracer(__name__)
```

#### 3. FastAPI Instrumentation
Instrument FastAPI application:

```python
# app/main.py
from utils.tracing import setup_tracing, get_tracer
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Set up tracing
setup_tracing("clstock-api")

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)
```

### Phase 2: Manual Tracing Implementation

#### 1. API Endpoint Tracing
Add manual tracing to key API endpoints:

```python
# api/endpoints.py
from utils.tracing import get_tracer
from opentelemetry import trace

@router.get("/recommendations")
async def get_recommendations(top_n: int = Query(5, ge=1, le=10)):
    tracer = get_tracer()
    
    with tracer.start_as_current_span("get_recommendations") as span:
        span.set_attribute("top_n", top_n)
        
        try:
            predictor = StockPredictor()
            recommendations = predictor.get_top_recommendations(top_n)
            
            span.set_attribute("recommendation_count", len(recommendations))
            span.set_status(trace.Status(trace.StatusCode.OK))
            
            return recommendations
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
```

#### 2. Prediction Workflow Tracing
Add tracing to prediction workflows:

```python
# models/predictor.py
from utils.tracing import get_tracer

class StockPredictor:
    def calculate_score(self, symbol: str) -> float:
        tracer = get_tracer()
        
        with tracer.start_as_current_span("calculate_score") as span:
            span.set_attribute("symbol", symbol)
            
            try:
                score = self._calculate_score_impl(symbol)
                
                span.set_attribute("score", score)
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return score
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
```

#### 3. Data Fetching Tracing
Add tracing to data fetching operations:

```python
# data/stock_data.py
from utils.tracing import get_tracer

class StockDataProvider:
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        tracer = get_tracer()
        
        with tracer.start_as_current_span("get_stock_data") as span:
            span.set_attribute("symbol", symbol)
            span.set_attribute("period", period)
            
            try:
                # Add ticker information for Japanese stocks
                if symbol in self.jp_stock_codes:
                    ticker = f"{symbol}.T"
                else:
                    ticker = symbol
                
                span.set_attribute("ticker", ticker)
                
                logger.info(f"Fetching data for {symbol} (period: {period})")
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                
                span.set_attribute("data_points", len(data))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                if data.empty:
                    span.add_event("No data returned")
                
                return data
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
```

### Phase 3: Context Propagation

#### 1. Async Operation Context
Ensure context propagation in async operations:

```python
# data/async_stock_data.py
import asyncio
from opentelemetry import trace

async def get_stock_data_async(self, symbol: str, period: str = "1y") -> pd.DataFrame:
    tracer = get_tracer()
    
    with tracer.start_as_current_span("get_stock_data_async") as span:
        span.set_attribute("symbol", symbol)
        span.set_attribute("period", period)
        
        # Create task with context propagation
        task = asyncio.create_task(
            self._fetch_data_with_context(symbol, period, trace.get_current_span())
        )
        
        try:
            data = await task
            span.set_attribute("data_points", len(data))
            span.set_status(trace.Status(trace.StatusCode.OK))
            return data
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

async def _fetch_data_with_context(self, symbol: str, period: str, parent_span):
    # Recreate context for async task
    ctx = trace.set_span_in_context(parent_span)
    tracer = get_tracer()
    
    with tracer.start_span("fetch_data_task", context=ctx) as span:
        span.set_attribute("symbol", symbol)
        # Implementation here
```

#### 2. Cross-Service Context
Propagate context in external service calls:

```python
# utils/network_manager.py
import requests
from opentelemetry import trace
from opentelemetry.propagate import inject

def make_traced_request(url: str, method: str = "GET", **kwargs) -> requests.Response:
    """Make HTTP request with trace context propagation."""
    # Inject trace context into headers
    headers = kwargs.get("headers", {})
    inject(headers)
    kwargs["headers"] = headers
    
    tracer = get_tracer()
    with tracer.start_as_current_span(f"http_{method.lower()}") as span:
        span.set_attribute("http.url", url)
        span.set_attribute("http.method", method)
        
        try:
            response = requests.request(method, url, **kwargs)
            span.set_attribute("http.status_code", response.status_code)
            span.set_status(trace.Status(trace.StatusCode.OK))
            return response
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
```

## Tracing Backend Configuration

### Jaeger Setup
Configuration for Jaeger all-in-one:

```yaml
# docker-compose.yml
version: '3'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "14268:14268"  # HTTP collector
      - "6831:6831/udp"  # Jaeger thrift compact
```

### Zipkin Alternative
Configuration for Zipkin:

```yaml
# docker-compose.yml
version: '3'
services:
  zipkin:
    image: openzipkin/zipkin:latest
    ports:
      - "9411:9411"  # UI
```

## Trace Analysis and Debugging

### Key Metrics to Monitor
1. **Latency Distribution**: p50, p90, p99 response times
2. **Error Rates**: Percentage of failed traces
3. **Throughput**: Traces per second
4. **Service Dependencies**: Call graphs between services

### Common Debugging Scenarios
1. **Slow API Responses**: Identify bottlenecks in request flow
2. **Prediction Delays**: Trace prediction workflow performance
3. **Data Fetching Issues**: Monitor external API performance
4. **Model Training Performance**: Track training job durations

## Implementation Timeline

### Week 1
- Set up OpenTelemetry dependencies
- Configure tracer provider and exporters
- Instrument FastAPI application

### Week 2
- Add manual tracing to API endpoints
- Implement tracing in prediction workflows
- Add tracing to data fetching operations

### Week 3
- Implement context propagation for async operations
- Add cross-service context propagation
- Configure tracing backend (Jaeger/Zipkin)

### Week 4
- Performance testing and optimization
- Documentation and training
- Production deployment

## Success Metrics

1. **Coverage**: >95% of critical workflows traced
2. **Performance**: <2% overhead on request processing
3. **Reliability**: >99.9% trace collection success rate
4. **Debuggability**: Reduction in mean time to resolve issues by 40%