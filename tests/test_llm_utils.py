from src.utils.llm import parse_llm_json


def test_parse_llm_json_handles_markdown_fence():
    text = """```json
    {"ok": true, "items": []}
    ```"""
    data = parse_llm_json(text)
    assert data["ok"] is True


def test_parse_llm_json_repairs_unterminated_string():
    text = """{
      "has_visual_content": true,
      "additional_text": "Line one
    }"""
    data = parse_llm_json(text)
    assert data["has_visual_content"] is True
    assert "Line one" in data["additional_text"]


def test_parse_llm_json_removes_trailing_comma_and_balances():
    text = """{
      "questions": ["a", "b",],
      "follow_ups": ["c"]
    """
    data = parse_llm_json(text)
    assert data["questions"] == ["a", "b"]
    assert data["follow_ups"] == ["c"]
