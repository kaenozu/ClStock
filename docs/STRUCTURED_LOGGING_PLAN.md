# Structured Logging Enhancement Plan

## Overview

This document outlines the implementation plan for enhancing the ClStock logging system with structured logging capabilities. The goal is to improve log searchability, analysis, and debugging capabilities through structured, contextual logging.

## Current State Analysis

The current logging system uses a custom logger configuration with basic formatting. While functional, it lacks:
- Structured data formatting (JSON)
- Contextual information in logs
- Standardized log levels and messages
- Easy correlation of related log entries
- Machine-readable log formats for analysis

## Enhancement Goals

1. **Structured Logging**: Output logs in JSON format with structured data
2. **Context Enrichment**: Add contextual information to log entries
3. **Correlation IDs**: Implement request tracing with correlation IDs
4. **Standardization**: Standardize log formats and levels
5. **Performance**: Maintain logging performance with minimal overhead

## Implementation Approach

### Phase 1: Structured Logging Library Integration

#### 1. Dependency Installation
Add structured logging library to requirements.txt:

```txt
structlog>=21.0.0
python-json-logger>=2.0.0
```

#### 2. Logger Configuration
Enhance the existing logger configuration to support structured logging:

```python
# utils/logger_config.py
import structlog
import logging
import sys
from typing import Dict, Any

def setup_structured_logger(name: str = None, level: int = logging.INFO) -> structlog.BoundLogger:
    """Set up structured logger with JSON formatting."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(name)

def bind_context(**kwargs):
    """Bind context to the current logger."""
    return structlog.get_logger().bind(**kwargs)
```

### Phase 2: Contextual Logging Implementation

#### 1. Request Context
Add request context to API logs:

```python
# api/endpoints.py
from utils.logger_config import setup_structured_logger, bind_context
import uuid

logger = setup_structured_logger(__name__)

@router.get("/recommendations")
async def get_recommendations(top_n: int = Query(5, ge=1, le=10)):
    # Generate correlation ID for request
    correlation_id = str(uuid.uuid4())
    
    # Bind context to logger
    ctx_logger = bind_context(
        correlation_id=correlation_id,
        endpoint="get_recommendations",
        top_n=top_n
    )
    
    ctx_logger.info("Processing recommendations request")
    
    try:
        predictor = StockPredictor()
        recommendations = predictor.get_top_recommendations(top_n)
        
        ctx_logger.info(
            "Recommendations generated successfully",
            recommendation_count=len(recommendations),
            symbols=[r.symbol for r in recommendations]
        )
        
        return recommendations
    except Exception as e:
        ctx_logger.error(
            "Failed to generate recommendations",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise
```

#### 2. Prediction Context
Add context to prediction logs:

```python
# models/predictor.py
from utils.logger_config import bind_context

class StockPredictor:
    def calculate_score(self, symbol: str) -> float:
        # Bind prediction context
        ctx_logger = bind_context(
            component="StockPredictor",
            method="calculate_score",
            symbol=symbol
        )
        
        ctx_logger.info("Starting score calculation")
        
        try:
            score = self._calculate_score_impl(symbol)
            
            ctx_logger.info(
                "Score calculated successfully",
                score=score,
                confidence=self._get_confidence_level()
            )
            
            return score
        except Exception as e:
            ctx_logger.error(
                "Score calculation failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
```

#### 3. Data Fetching Context
Add context to data fetching logs:

```python
# data/stock_data.py
from utils.logger_config import bind_context

class StockDataProvider:
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        # Bind data fetching context
        ctx_logger = bind_context(
            component="StockDataProvider",
            method="get_stock_data",
            symbol=symbol,
            period=period
        )
        
        ctx_logger.info("Fetching stock data")
        
        try:
            # Add ticker information for Japanese stocks
            if symbol in self.jp_stock_codes:
                ticker = f"{symbol}.T"
            else:
                ticker = symbol
            
            ctx_logger.info(
                "Fetching data from external source",
                ticker=ticker,
                source="yahoo_finance"
            )
            
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            ctx_logger.info(
                "Data fetched successfully",
                data_points=len(data),
                date_range=f"{data.index.min()} to {data.index.max()}" if not data.empty else "no data"
            )
            
            return data
        except Exception as e:
            ctx_logger.error(
                "Failed to fetch stock data",
                error_type=type(e).__name__,
                error_message=str(e),
                ticker=ticker
            )
            raise
```

### Phase 3: Correlation and Tracing

#### 1. Correlation ID Propagation
Ensure correlation IDs are propagated through async operations:

```python
# utils/context_manager.py
import structlog
import asyncio
from contextvars import ContextVar

# Context variable for correlation ID
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id_var.get()

def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    correlation_id_var.set(correlation_id)

async def with_correlation_id(correlation_id: str, coro):
    """Execute coroutine with correlation ID context."""
    token = correlation_id_var.set(correlation_id)
    try:
        return await coro
    finally:
        correlation_id_var.reset(token)
```

#### 2. Async Context Propagation
Propagate context in async operations:

```python
# data/async_stock_data.py
from utils.context_manager import with_correlation_id, get_correlation_id

async def get_stock_data_async(self, symbol: str, period: str = "1y") -> pd.DataFrame:
    correlation_id = get_correlation_id()
    
    # Create task with context propagation
    task = asyncio.create_task(
        with_correlation_id(
            correlation_id,
            self._fetch_data_async(symbol, period)
        )
    )
    
    return await task
```

### Phase 4: Log Analysis and Search

#### 1. Log Format Standardization
Standardize log formats for easy parsing:

```json
{
  "timestamp": "2023-01-01T10:00:00.000Z",
  "logger": "models.predictor",
  "level": "INFO",
  "message": "Score calculated successfully",
  "correlation_id": "123e4567-e89b-12d3-a456-426614174000",
  "component": "StockPredictor",
  "method": "calculate_score",
  "symbol": "7203",
  "score": 78.5,
  "confidence": 0.87
}
```

#### 2. Log Aggregation Setup
Configure log aggregation for analysis:

```python
# utils/log_aggregator.py
import json
from typing import Dict, Any
from pathlib import Path

class LogAggregator:
    def __init__(self, log_file: str = "logs/clstock_structured.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
    
    def write_log(self, log_entry: Dict[str, Any]) -> None:
        """Write structured log entry to file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def search_logs(self, **filters) -> list:
        """Search logs with filters."""
        matching_logs = []
        
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        if self._matches_filters(log_entry, filters):
                            matching_logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        
        return matching_logs
    
    def _matches_filters(self, log_entry: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if log entry matches filters."""
        for key, value in filters.items():
            if key not in log_entry or log_entry[key] != value:
                return False
        return True
```

## Integration with Existing System

### Backward Compatibility
Maintain backward compatibility with existing logging:

```python
# utils/logger_config.py
import logging
import structlog
from typing import Union

def get_logger(name: str = None, structured: bool = True) -> Union[logging.Logger, structlog.BoundLogger]:
    """Get logger instance, either structured or traditional."""
    if structured:
        return setup_structured_logger(name)
    else:
        return setup_logger(name)
```

### Configuration Options
Add configuration for logging options:

```python
# config/settings.py
@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    structured: bool = True  # Enable structured logging
    log_file: str = "logs/clstock.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
```

## Implementation Timeline

### Week 1
- Set up structured logging dependencies
- Configure logger with JSON formatting
- Implement basic context binding

### Week 2
- Add contextual logging to API endpoints
- Implement contextual logging in prediction workflows
- Add contextual logging to data fetching operations

### Week 3
- Implement correlation ID propagation
- Add async context propagation
- Set up log aggregation and search capabilities

### Week 4
- Performance testing and optimization
- Documentation and training
- Production deployment

## Success Metrics

1. **Structured Logs**: 100% of new logs in JSON format
2. **Context Enrichment**: >90% of log entries contain contextual information
3. **Searchability**: Log search time reduced by 70%
4. **Debugging**: Mean time to debug issues reduced by 50%
5. **Performance**: <1% overhead on application performance