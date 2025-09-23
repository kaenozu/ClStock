# Code Duplication Analysis and Refactoring Plan

## Identified Duplications

1. **predict_return_rate method** - Exists in multiple classes:
   - `models/predictor.py` - StockPredictor class
   - `models/ml_models.py` - MLStockPredictor class
   - `models/core.py` - MLStockPredictor class
   - `research/practical_predictor.py` - PracticalPredictor class

2. **Similar technical indicator calculations** - Found in multiple files

3. **Common data processing patterns** - Repeated across predictor classes

## Refactoring Plan

### Step 1: Create Shared Base Classes

Create a common base class that all predictors can inherit from:

```python
# models/base.py
class BaseStockPredictor(ABC):
    """Base class for all stock prediction models."""
    
    def __init__(self):
        self.data_provider = StockDataProvider()
        self.settings = get_settings()
    
    def predict_return_rate(self, symbol: str, days: int = 5) -> float:
        """Common implementation for return rate prediction.
        
        Args:
            symbol: Stock symbol to predict
            days: Prediction horizon in days
            
        Returns:
            Predicted return rate as decimal
        """
        # Common implementation that can be overridden by subclasses
        pass
        
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard technical indicators."""
        return self.data_provider.calculate_technical_indicators(data)
```

### Step 2: Create Shared Utility Modules

Create utility modules for common functionality:

```python
# utils/predictors.py
def standardize_return_rate_prediction(symbol: str, data_provider, settings) -> float:
    """Standardized return rate prediction logic."""
    # Implementation here
    
def calculate_prediction_confidence(prediction: float, model_accuracy: float) -> float:
    """Calculate confidence level for a prediction."""
    # Implementation here
```

### Step 3: Refactor Existing Classes

Update existing predictor classes to use shared components:

```python
# models/predictor.py
class StockPredictor(BaseStockPredictor):
    def predict_return_rate(self, symbol: str) -> float:
        """Predict return rate using rule-based approach."""
        # Use shared utilities and base class methods
        return super().predict_return_rate(symbol, days=5)
```

## Benefits

1. **Reduced Code Duplication**: Single source of truth for common functionality
2. **Easier Maintenance**: Changes to shared logic only need to be made in one place
3. **Improved Consistency**: All predictors will use the same underlying implementations
4. **Better Testability**: Shared components can be tested independently