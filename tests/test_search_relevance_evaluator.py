"""Pure tests for the product-search relevance gate."""

from scripts.evaluate_product_search import CASES, RelevanceCase, evaluate_case


def test_evaluator_accepts_exact_structured_results():
    case = RelevanceCase(
        name="mixed",
        query="black Kowtow jeans",
        top_n=2,
        min_results=2,
        allowed_categories=("jeans",),
        allowed_colors=("black",),
        brand_prefix="kowtow",
    )
    results = [
        {
            "product_title": "Classic Jeans",
            "product_brand": "Kowtowclothing",
            "category": "jeans",
            "product_color": "black",
        },
        {
            "product_title": "High Puddle Jeans",
            "product_brand": "Kowtow Clothing",
            "category": "jeans",
            "product_color": "Black.",
        },
    ]

    report = evaluate_case(case, results)

    assert report["passed"] is True
    assert report["failures"] == []


def test_evaluator_reports_missing_and_conflicting_facets():
    case = RelevanceCase(
        name="color_category",
        query="black jeans",
        top_n=3,
        min_results=3,
        allowed_categories=("jeans",),
        allowed_colors=("black",),
    )
    results = [
        {"product_title": "Blue Jeans", "category": "jeans", "product_color": "blue"},
        {"product_title": "Mystery Jeans", "category": "jeans"},
        {"product_title": "Black Boots", "category": "boot", "product_color": "black"},
    ]

    report = evaluate_case(case, results)

    assert report["passed"] is False
    assert "unexpected categories: boot" in report["failures"]
    assert "1 results are missing product_color" in report["failures"]
    assert "unexpected colors: blue" in report["failures"]


def test_pants_contract_requires_jeans_and_trouser_language():
    pants_case = CASES[0]
    results = [
        {"product_title": "Relaxed Pant", "category": "pants"},
        {"product_title": "Classic Jeans", "category": "jeans"},
        {"product_title": "Everyday Leggings", "category": "leggings"},
    ] * 4

    report = evaluate_case(pants_case, results)

    assert report["passed"] is False
    assert "missing title family: trouser / slack / chino" in report["failures"]
