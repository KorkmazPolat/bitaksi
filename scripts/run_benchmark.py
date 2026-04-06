from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import chromadb

from src.api.dependencies import get_chat_service
from src.config import get_settings


BENCHMARK_PATH = Path("docs/benchmarks/bitaksi_benchmark_50.json")
RESULTS_PATH = Path("docs/benchmarks/bitaksi_benchmark_50_results.json")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_doc_map() -> dict[str, str]:
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    raw = client.get_collection(name=settings.chroma_collection_raw)
    records = raw.get(include=["metadatas"])
    doc_ids = sorted({(m or {}).get("doc_id", "") for m in records.get("metadatas", []) if (m or {}).get("doc_id")})

    resolved: dict[str, str] = {}
    normalized = {doc_id: normalize_text(doc_id) for doc_id in doc_ids}

    for doc_id, norm in normalized.items():
        if "performans" in norm:
            resolved["performans_kilavuzu_2025"] = doc_id
        elif "izin" in norm:
            resolved["izin_proseduru"] = doc_id
        elif "masraf" in norm:
            resolved["masraf_uygulamasi"] = doc_id
        elif "seyahat" in norm:
            resolved["seyahat_proseduru"] = doc_id
        elif "zimmetli" in norm:
            resolved["zimmetli_esya_proseduru"] = doc_id

    return resolved


def expected_aliases_to_doc_ids(expected_docs: list[str], doc_map: dict[str, str]) -> list[str]:
    out: list[str] = []
    known_doc_ids = set(doc_map.values())
    for value in expected_docs:
        if value in doc_map:
            out.append(doc_map[value])
        elif value in known_doc_ids:
            out.append(value)
    return out


def looks_like_abstain(answer: str) -> bool:
    norm = normalize_text(answer)
    patterns = [
        "bilgi bulunmamaktadir",
        "yer almamaktadir",
        "dair bilgi bulunmamaktadir",
        "dogrudan insan kaynaklari",
        "ik departmani",
        "bu konuda bilgi almak",
    ]
    return any(pattern in norm for pattern in patterns)


def main() -> None:
    with BENCHMARK_PATH.open() as f:
        benchmark = json.load(f)

    doc_map = load_doc_map()
    service = get_chat_service()

    items_out: list[dict] = []
    strategy_counter: Counter[str] = Counter()

    for item in benchmark["items"]:
        result = service.answer(query=item["question"], history=[])
        source_docs = [source["document"] for source in result.sources]
        expected_doc_ids = expected_aliases_to_doc_ids(item["expected_docs"], doc_map)
        expected_doc_match = bool(set(source_docs) & set(expected_doc_ids))

        answerable_expected = item["expected_outcome"] == "answerable"
        abstain_expected = item["expected_outcome"] == "abstain"
        abstain_text = looks_like_abstain(result.answer or "")
        behaved_correctly = (
            (answerable_expected and result.grounded)
            or (abstain_expected and not result.grounded)
        )
        safe_behavior_correct = (
            (answerable_expected and result.grounded)
            or (abstain_expected and abstain_text)
        )

        if answerable_expected:
            doc_behavior_ok = result.grounded and expected_doc_match
        else:
            doc_behavior_ok = abstain_text

        strategy_counter[str(result.strategy_used)] += 1

        items_out.append(
            {
                "id": item["id"],
                "category": item["category"],
                "expected_outcome": item["expected_outcome"],
                "question": item["question"],
                "expected_docs_aliases": item["expected_docs"],
                "expected_doc_ids": expected_doc_ids,
                "grounded": result.grounded,
                "strategy_used": str(result.strategy_used),
                "queries_tried": result.queries_tried,
                "source_docs": source_docs,
                "source_count": len(result.sources),
                "expected_doc_match": expected_doc_match,
                "behavior_correct": behaved_correctly,
                "safe_behavior_correct": safe_behavior_correct,
                "doc_behavior_correct": doc_behavior_ok,
                "abstain_text_detected": abstain_text,
                "answer_preview": (result.answer or "")[:400],
            }
        )

    answerable = [x for x in items_out if x["expected_outcome"] == "answerable"]
    abstain = [x for x in items_out if x["expected_outcome"] == "abstain"]

    summary = {
        "num_questions": len(items_out),
        "num_answerable": len(answerable),
        "num_abstain": len(abstain),
        "grounded_rate": round(sum(1 for x in items_out if x["grounded"]) / max(len(items_out), 1), 4),
        "answerable_success_rate": round(sum(1 for x in answerable if x["behavior_correct"]) / max(len(answerable), 1), 4),
        "abstain_accuracy": round(sum(1 for x in abstain if x["behavior_correct"]) / max(len(abstain), 1), 4),
        "safe_abstain_accuracy": round(sum(1 for x in abstain if x["safe_behavior_correct"]) / max(len(abstain), 1), 4),
        "expected_doc_match_rate_answerable": round(sum(1 for x in answerable if x["expected_doc_match"]) / max(len(answerable), 1), 4),
        "full_behavior_score_answerable": round(sum(1 for x in answerable if x["doc_behavior_correct"]) / max(len(answerable), 1), 4),
        "strategy_distribution": dict(strategy_counter),
        "doc_map": doc_map,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump({"summary": summary, "items": items_out}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
