from api.schemas import EntryPriceSchema, StockRecommendationSchema, TargetSchema


def test_stock_recommendation_schema_includes_company_name_key():
    schema = StockRecommendationSchema(
        rank=1,
        symbol="7203",
        company_name="Test Corp",
        sector=None,
        score=85.0,
        action="buy",
        action_text="Buy dips",
        entry_condition="Pullback",
        entry_price=EntryPriceSchema(min=98.0, max=102.0),
        stop_loss=95.0,
        targets=[
            TargetSchema(label="base", price=110.0),
            TargetSchema(label="stretch", price=120.0),
        ],
        holding_period_days=30,
        confidence=0.75,
        rationale="Strong fundamentals",
        notes=None,
        risk_level="medium",
        chart_refs=None,
        current_price=100.0,
    )

    dumped = schema.model_dump()
    assert dumped["company_name"] == "Test Corp"
    assert "name" not in dumped
