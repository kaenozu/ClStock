# ClStock API Documentation

## Overview

The ClStock API provides access to mid-term stock recommendations for Japanese equities. All endpoints require authentication via API key and are subject to rate limiting.

## Authentication

All API endpoints require a valid API key passed in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

Available API keys:
- Development: `development-key`
- Administration: `admin-key`

## Rate Limits

- `/recommendations`: 50 requests per minute
- `/recommendation/{symbol}`: 100 requests per minute
- `/stocks`: 200 requests per minute
- `/stock/{symbol}/data`: 150 requests per minute

## Endpoints

### Get Stock Recommendations

#### `GET /api/v1/recommendations`

Get top recommended stocks for mid-term investment (30-90 days).

**Query Parameters:**
- `top_n` (integer, optional): Number of recommendations to return (1-10, default: 5)

**Response:** (conforms to `api.schemas.RecommendationResponse`)
```json
{
  "recommendations": [
    {
      "rank": 1,
      "symbol": "7203",
      "company_name": "トヨタ自動車",
      "buy_timing": "昨日の高値（3,250円）を超えたら買い",
      "target_price": 3510.0,
      "stop_loss": 3152.0,
      "profit_target_1": 3442.0,
      "profit_target_2": 3577.0,
      "holding_period": "1～2か月",
      "score": 78.5,
      "current_price": 3200.0,
      "recommendation_reason": "強い上昇トレンドとファンダメンタルズが良好"
    }
  ],
  "generated_at": "2023-01-01T10:00:00",
  "market_status": "市場営業時間外"
}
```

**Example Request:**
```bash
curl -H "Authorization: Bearer development-key" "http://localhost:8000/api/v1/recommendations?top_n=3"
```

### Get Single Stock Recommendation

#### `GET /api/v1/recommendation/{symbol}`

Get detailed recommendation for a specific stock.

**Path Parameters:**
- `symbol` (string, required): Stock symbol (e.g., "7203")

**Response:**
```json
{
  "rank": 1,
  "symbol": "7203",
  "company_name": "トヨタ自動車",
  "buy_timing": "昨日の高値（3,250円）を超えたら買い",
  "target_price": 3510.0,
  "stop_loss": 3152.0,
  "profit_target_1": 3442.0,
  "profit_target_2": 3577.0,
  "holding_period": "1～2か月",
  "score": 78.5,
  "current_price": 3200.0,
  "recommendation_reason": "強い上昇トレンドとファンダメンタルズが良好"
}
```

**Example Request:**
```bash
curl -H "Authorization: Bearer development-key" "http://localhost:8000/api/v1/recommendation/7203"
```

### Get Available Stocks

#### `GET /api/v1/stocks`

Get list of all available stocks in the system.

**Response:**
```json
{
  "stocks": [
    {
      "symbol": "7203",
      "name": "トヨタ自動車"
    },
    {
      "symbol": "6758",
      "name": "ソニーグループ"
    }
  ]
}
```

**Example Request:**
```bash
curl -H "Authorization: Bearer development-key" "http://localhost:8000/api/v1/stocks"
```

### Get Stock Data

#### `GET /api/v1/stock/{symbol}/data`

Get detailed stock data including technical indicators and financial metrics.

**Path Parameters:**
- `symbol` (string, required): Stock symbol (e.g., "7203")

**Query Parameters:**
- `period` (string, optional): Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max", default: "1y")

**Response:**
```json
{
  "symbol": "7203",
  "company_name": "トヨタ自動車",
  "current_price": 3200.0,
  "price_change": 50.0,
  "price_change_percent": 1.59,
  "volume": 1000000,
  "technical_indicators": {
    "sma_20": 3150.0,
    "sma_50": 3100.0,
    "rsi": 65.0,
    "macd": 25.0
  },
  "financial_metrics": {
    "market_cap": 1800000000000,
    "pe_ratio": 12.5,
    "dividend_yield": 2.1
  },
  "last_updated": "2023-01-01T10:00:00"
}
```

**Example Request:**
```bash
curl -H "Authorization: Bearer development-key" "http://localhost:8000/api/v1/stock/7203/data?period=6mo"
```

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message"
}
```

Common HTTP status codes:
- `400`: Bad Request - Invalid parameters
- `401`: Unauthorized - Missing or invalid API key
- `404`: Not Found - Requested resource not found
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Unexpected server error

## SDKs and Libraries

Currently, there are no official SDKs. API requests can be made using standard HTTP libraries in any programming language.

## Versioning

The API is currently at version 1.0.0. Breaking changes will result in a new version path (e.g., `/api/v2/`).