# Naming Convention Standards

## Function Naming

### General Principles
1. **Use descriptive names** that clearly indicate what the function does
2. **Follow verb-noun pattern** for action functions (e.g., `calculate_score`, `get_stock_data`)
3. **Use boolean naming** for functions that return boolean values (e.g., `is_trained`, `has_data`)
4. **Consistent abbreviations** - Use standard abbreviations consistently:
   - `sma` for Simple Moving Average
   - `rsi` for Relative Strength Index
   - `macd` for Moving Average Convergence Divergence
   - `pct` for percentage
   - `vol` for volume
   - `idx` for index
   - `df` for DataFrame (in type hints only)

### Specific Patterns

#### Calculation Functions
- Pattern: `calculate_[what_is_calculated]`
- Examples:
  - `calculate_score(symbol: str) -> float`
  - `calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame`
  - `calculate_position_size(symbol: str, confidence: float) -> float`

#### Getter Functions
- Pattern: `get_[what_is_returned]`
- Examples:
  - `get_stock_data(symbol: str, period: str) -> pd.DataFrame`
  - `get_multiple_stocks(symbols: List[str], period: str) -> Dict[str, pd.DataFrame]`
  - `get_confidence() -> float`

#### Prediction Functions
- Pattern: `predict_[what_is_predicted]`
- Examples:
  - `predict_return_rate(symbol: str) -> float`
  - `predict_price_target(symbol: str) -> Tuple[float, float]`
  - `predict_direction(symbol: str) -> Dict[str, float]`

#### Helper/Utility Functions
- Pattern: `[action]_[target]` or `[action]_[target]_[specification]`
- Examples:
  - `prepare_features(data: pd.DataFrame) -> pd.DataFrame`
  - `validate_input(symbol: str) -> bool`
  - `format_response(data: Dict) -> str`

## Variable Naming

### General Principles
1. **Be descriptive** - Variable names should clearly indicate their purpose
2. **Use full words** instead of cryptic abbreviations
3. **Consistent context** - Use the same names for the same concepts across the codebase

### Common Variables
- `symbol`: Stock symbol (e.g., "7203")
- `data`: DataFrame containing stock data
- `score`: Numerical score value
- `price`: Price value
- `returns`: Series of return values
- `features`: DataFrame of feature variables
- `predictions`: List or array of predictions
- `confidence`: Confidence level (0.0-1.0)

## Class Naming

### General Principles
1. **Use PascalCase** for all class names
2. **Use noun phrases** that describe what the class represents
3. **Be specific** about the class's purpose

### Common Patterns
- `[Noun]` for general classes (e.g., `StockPredictor`, `DataProcessor`)
- `[Noun][Verb]` for specialized classes (e.g., `EnsemblePredictor`, `CacheablePredictor`)
- `[Adjective][Noun]` for classes with specific characteristics (e.g., `AdvancedEnsemblePredictor`)

## Module Naming

### General Principles
1. **Use snake_case** for all module names
2. **Be descriptive** but concise
3. **Use plural forms** for modules containing collections (e.g., `models`, `utils`)
4. **Use singular forms** for modules representing a single concept (e.g., `predictor`, `settings`)

## Constants Naming

### General Principles
1. **Use UPPER_SNAKE_CASE** for all constants
2. **Be explicit** about what the constant represents
3. **Group related constants** logically

### Examples
- `DEFAULT_CONFIDENCE_THRESHOLD = 0.8`
- `MAX_PREDICTION_DAYS = 30`
- `MIN_TRAINING_DATA_POINTS = 100`

## Type Hinting Standards

### General Principles
1. **Always include type hints** for function parameters and return values
2. **Use specific types** when possible (e.g., `List[str]` instead of `list`)
3. **Import typing modules** at the top of files
4. **Use Union types** for variables that can have multiple types

### Common Type Patterns
- `symbol: str` - Stock symbol
- `data: pd.DataFrame` - Stock data DataFrame
- `symbols: List[str]` - List of stock symbols
- `result: Dict[str, Any]` - Generic result dictionary
- `optional_data: Optional[pd.DataFrame]` - Data that may be None
- `confidence: float` - Confidence value between 0.0 and 1.0

## Configuration Parameter Naming

### General Principles
1. **Use descriptive names** that clearly indicate the parameter's purpose
2. **Follow category_parameter pattern** for organized grouping
3. **Use consistent units** (e.g., percentages as decimals, time in seconds)

### Examples
- `trading_max_position_size: float` - Maximum position size as decimal
- `model_training_epochs: int` - Number of training epochs
- `data_cache_ttl_seconds: int` - Cache time-to-live in seconds
- `api_rate_limit_requests: int` - Number of requests for rate limiting

## Error and Exception Naming

### General Principles
1. **End with "Error"** for exception classes
2. **Be specific** about the type of error
3. **Inherit from appropriate base classes**

### Examples
- `DataFetchError` - Error fetching data
- `InvalidSymbolError` - Invalid stock symbol provided
- `ModelTrainingError` - Error during model training
- `PredictionError` - Error during prediction