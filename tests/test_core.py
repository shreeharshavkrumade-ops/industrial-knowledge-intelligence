"""Unit + integration tests. Run:  python -m pytest -q"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import storage  # noqa: E402
from src.compliance_checker import run_compliance, summarize  # noqa: E402
from src.document_processor import process_bytes  # noqa: E402
from src.entity_extractor import extract_from_sentence  # noqa: E402
from src.evaluation import evaluate_entities, evaluate_retrieval, precision_at_k, prf, recall_at_k, reciprocal_rank  # noqa: E402
from src.glass_component import ASSET_DIR, build_header_html  # noqa: E402
from src.knowledge_assistant import answer_question  # noqa: E402
from src.maintenance_analyzer import cluster_faults, prepare_maintenance, train_risk_model  # noqa: E402
from src.pipeline import KnowledgeBase  # noqa: E402
from src.retrieval_engine import RetrievalEngine  # noqa: E402
from src.text_preprocessor import chunk_text, clean_text, split_sentences  # noqa: E402

_ORIG_INSIDE_PROJECT = storage._inside_project  # captured before fixtures monkeypatch it


# ---------------- preprocessing ----------------
def test_clean_text_normalises_whitespace_and_wraps():
    raw = "Pump  P-101\u00a0is  a pump\nthat runs\n\n\n\nNext   paragraph\x00."
    out = clean_text(raw)
    assert "  " not in out and "\x00" not in out and "\n\n\n" not in out
    assert "is a pump that runs" in out  # wrapped line joined


def test_clean_text_handles_missing_values():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_chunking_overlaps_and_covers_text():
    text = " ".join(f"Sentence number {i} about pump bearing maintenance." for i in range(60))
    chunks = chunk_text(text, chunk_words=40, overlap_words=12)
    assert len(chunks) > 3
    for a, b in zip(chunks, chunks[1:]):
        assert split_sentences(a)[-1] in b  # last sentence of a chunk is repeated at the start of the next
    assert "Sentence number 59" in chunks[-1]


def test_chunking_rejects_bad_parameters():
    with pytest.raises(ValueError):
        chunk_text("abc def", chunk_words=10, overlap_words=10)


# ---------------- ingestion ----------------
def test_process_txt_and_page_info():
    doc, chunks = process_bytes("a.txt", b"Pump P-101 had a bearing failure. It was replaced.")
    assert doc.status == "processed" and doc.n_chunks == len(chunks) >= 1 and chunks[0]["document"] == "a.txt"


@pytest.mark.parametrize("name,data,status", [("e.txt", b"", "empty"), ("x.exe", b"abc", "unsupported"),
                                              ("w.txt", b"   \n  ", "empty"), ("bad.pdf", b"not a pdf", "error"),
                                              ("bad.docx", b"not a docx", "error")])
def test_bad_inputs_never_raise(name, data, status):
    doc, chunks = process_bytes(name, data)
    assert doc.status == status and chunks == [] and doc.message


def test_pdf_keeps_page_numbers():
    doc, chunks = process_bytes("boiler_b301_inspection_report.pdf", (ROOT / "data/sample_documents/boiler_b301_inspection_report.pdf").read_bytes())
    assert doc.n_pages == 2 and {c["page"] for c in chunks} == {1, 2} and chunks[0]["location"] == "page 1"


def test_docx_tables_are_extracted():
    doc, _ = process_bytes("m.docx", (ROOT / "data/sample_documents/motor_inspection_safety_procedure.docx").read_bytes())
    assert "Stored electrical energy" in " ".join(p[2] for p in doc.pages)


def test_csv_rows_become_chunks():
    doc, chunks = process_bytes("r.csv", b"equipment_id,fault\nP-101,leak\nC-202,overheating\n")
    assert len(chunks) == 2 and chunks[1]["location"] == "row 2"


# ---------------- entity extraction ----------------
def _types(sentence):
    return {(e["entity_type"], e["value"]) for e in extract_from_sentence(sentence)}


def test_entities_asset_reg_and_inferred_type():
    ents = extract_from_sentence("Lock out Motor M-401 before work on Pump P-101 per IKI-SAF-007.")
    vals = {(e["entity_type"], e["value"]): e["kind"] for e in ents}
    assert ("asset_id", "M-401") in vals and ("asset_id", "P-101") in vals and ("regulatory_reference", "IKI-SAF-007") in vals
    assert vals[("equipment_type", "Pump")] == "inferred" and vals[("asset_id", "P-101")] == "extracted"


def test_negation_and_longest_match():
    assert ("fault", "corrosion") not in _types("No corrosion was found on the impeller.")
    t = _types("The mechanical seal leakage was repaired.")
    assert ("fault", "seal leakage") in t and ("fault", "leakage") not in t and ("component", "mechanical seal") in t


def test_work_order_id_is_not_asset_id():
    assert not any(t == "asset_id" for t, _ in _types("See WO-2026-0101 and IKI-MAN-P101."))


# ---------------- retrieval + metrics ----------------
@pytest.fixture(scope="module")
def tiny_engine():
    texts = {"pump.txt": "Pump P-101 bearing failure replaced coupling realigned", "comp.txt": "Compressor C-202 overheating oil cooler fouled",
             "boiler.txt": "Boiler B-301 tube fouling scale corrosion inspection", "motor.txt": "Motor M-401 insulation resistance megger test winding",
             "safety.txt": "Lockout tagout isolate energy before motor inspection wear gloves"}
    ch = pd.DataFrame([{"doc_id": f"d{i}", "document": k, "chunk_id": f"c{i}", "location": "-", "text": v, "file_type": "txt", "n_words": 8} for i, (k, v) in enumerate(texts.items())])
    return RetrievalEngine(ch, lsa_components=3)


def test_tfidf_finds_right_document_and_ids(tiny_engine):
    assert tiny_engine.search("overheating compressor", "tfidf", 1).iloc[0]["document"] == "comp.txt"
    assert tiny_engine.search("P-101", "tfidf", 1).iloc[0]["document"] == "pump.txt"
    assert tiny_engine.search("zzzz qqqq", "tfidf", 3).empty  # no fake matches


def test_semantic_mode_never_faked(tiny_engine):
    ok, _ = tiny_engine.availability()["semantic"]
    if not ok:
        with pytest.raises(RuntimeError):
            tiny_engine.scores("anything", "semantic")


def test_rank_metrics_hand_computed():
    ranked, rel = ["a", "x", "b", "y"], {"a", "b"}
    assert precision_at_k(ranked, rel, 2) == 0.5 and recall_at_k(ranked, rel, 3) == 1.0
    assert reciprocal_rank(["x", "b"], {"b"}) == 0.5 and reciprocal_rank(["x"], {"b"}) == 0.0


def test_evaluate_retrieval_matches_manual(tiny_engine):
    bench = pd.DataFrame({"query": ["overheating compressor", "motor insulation test"], "relevant_documents": ["comp.txt", "motor.txt"]})
    r = evaluate_retrieval(tiny_engine, bench, modes=["tfidf"], ks=(1,))
    assert r["summary"].iloc[0]["MRR"] == 1.0 and r["summary"].iloc[0]["P@1"] == 1.0


def test_prf_and_entity_evaluation_exact():
    assert prf(2, 2, 0) == (0.5, 1.0, pytest.approx(2 / 3)) and prf(0, 0, 0) == (0.0, 0.0, 0.0)
    ext = pd.DataFrame({"document": ["d"] * 3, "entity_type": ["fault", "fault", "asset_id"], "value": ["cavitation", "noise", "p-101"]})
    gt = pd.DataFrame({"document": ["d"] * 3, "entity_type": ["fault", "fault", "asset_id"], "value": ["cavitation", "vibration", "P-101"]})
    r = evaluate_entities(ext, gt)
    f = r["per_type"].set_index("entity_type")
    assert (f.loc["fault", "TP"], f.loc["fault", "FP"], f.loc["fault", "FN"]) == (1, 1, 1) and f.loc["asset_id", "f1"] == 1.0  # case-normalised
    assert len(r["errors"]) == 2


# ---------------- maintenance analytics ----------------
@pytest.fixture(scope="module")
def maint():
    return prepare_maintenance(pd.read_csv(ROOT / "data/maintenance_records.csv"))[0]


def test_prepare_maintenance_imputes_and_validates(maint):
    assert maint["downtime_hours"].notna().all() and pd.api.types.is_datetime64_any_dtype(maint["date"])
    with pytest.raises(ValueError):
        prepare_maintenance(pd.DataFrame({"a": [1]}))


def test_clustering_is_reproducible(maint):
    a, b = cluster_faults(maint), cluster_faults(maint)
    assert (a["data"]["cluster"].values == b["data"]["cluster"].values).all()
    assert -1 <= a["metrics"]["silhouette"] <= 1 and a["metrics"]["k"] >= 2


def test_classifier_split_and_no_leakage_columns(maint):
    r = train_risk_model(maint)
    assert r["available"] and r["n_train"] + r["n_test"] == len(maint)
    assert abs(r["train_positive_rate"] - r["test_positive_rate"]) < 0.08  # stratified
    assert r["target"] not in r["features"] and set(r["features"]).isdisjoint({"fault_label", "severity", "downtime_hours"})
    assert r["metrics"].equals(train_risk_model(maint)["metrics"])  # fixed seed => identical
    assert r["confusion_matrices"]["Random Forest"].sum() == r["n_test"]


def test_classifier_refuses_tiny_data(maint):
    assert train_risk_model(maint.head(20))["available"] is False


# ---------------- compliance ----------------
def test_compliance_statuses():
    req = pd.DataFrame([dict(requirement_id="R1", title="Vib", requirement_text="", check_type="periodic_inspection", applies_to="Pump", inspection_type="Vibration Analysis", max_interval_days=90, evidence_keywords=""),
                        dict(requirement_id="R2", title="Follow", requirement_text="", check_type="follow_up", applies_to="All", inspection_type="", max_interval_days=14, evidence_keywords=""),
                        dict(requirement_id="R3", title="LOTO", requirement_text="", check_type="documented_procedure", applies_to="All", inspection_type="", max_interval_days=0, evidence_keywords="lockout")])
    insp = pd.DataFrame([dict(inspection_id="I1", date="2026-01-01", equipment_id="P-1", inspection_type="Vibration Analysis", status="Fail", finding="bad"),
                         dict(inspection_id="I2", date="2026-09-01", equipment_id="P-2", inspection_type="Vibration Analysis", status="Pass", finding="ok")])
    assets = pd.DataFrame({"equipment_id": ["P-1", "P-2", "P-3"], "equipment_type": ["Pump"] * 3})
    res = run_compliance(req, insp, pd.DataFrame(), assets, pd.DataFrame(), date(2026, 9, 30)).set_index(["requirement_id", "equipment_id"])
    assert res.loc[("R1", "P-1"), "status"] == "Overdue" and res.loc[("R1", "P-2"), "status"] == "Compliant" and res.loc[("R1", "P-3"), "status"] == "Missing evidence"
    assert res.loc[("R2", "P-1"), "status"] == "Documentation gap" and res.loc[("R3", "(plant-wide)"), "status"] == "Missing evidence"
    assert summarize(res.reset_index())["gaps"] == 4


# ---------------- integration: ingestion -> models -> evaluation ----------------
@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("iki")
    mp.setattr(storage, "DB_PATH", tmp / "test.db"); mp.setattr(storage, "UPLOAD_DIR", tmp / "uploads")
    mp.setattr(storage, "_inside_project", lambda p: Path(p).resolve())
    k = KnowledgeBase(); k.load_samples()
    yield k
    mp.undo()


def test_pipeline_end_to_end(kb):
    s = kb.stats()
    assert s["documents"] == 11 and s["chunks"] > 100 and s["assets"] == 8 and s["incidents"] == 120
    assert {"asset_id", "fault", "component", "maintenance_action"} <= set(kb.entities["entity_type"])
    ev = kb.evaluate()
    assert ev["entity"]["available"] and ev["retrieval"]["available"] and ev["classification"]["available"] and len(ev["metrics_table"]) > 30
    assert 0 < ev["retrieval"]["summary"].iloc[0]["MRR"] <= 1


def test_duplicate_and_persistence(kb):
    data = (ROOT / "data/sample_documents/pump_p102_inspection_notes.txt").read_bytes()
    assert kb.ingest("copy_of_notes.txt", data)["status"] == "duplicate"
    assert KnowledgeBase().documents.shape[0] == 11  # reloaded from SQLite, not recomputed


def test_upload_is_saved_without_overwrite(kb):
    r1 = kb.ingest("new.txt", b"Pump P-102 gasket leakage tightened.")
    r2 = kb.ingest("new.txt", b"Different text about Boiler B-302 inspection.")
    kb.rebuild()
    assert r1["status"] == r2["status"] == "processed"
    assert len(list(storage.UPLOAD_DIR.glob("new*.txt"))) == 2


def test_assistant_is_grounded(kb):
    ok = answer_question(kb.engine, "What are the causes of overheating in Compressor C-202?")
    assert ok["sufficient"] and "fouled oil cooler" in ok["answer"] and ok["citations"]
    assert not answer_question(kb.engine, "What is the rated power of Pump P-999?")["sufficient"]
    assert not answer_question(kb.engine, "Who won the football world cup?")["sufficient"]


def test_assistant_expands_headings_and_has_no_duplicates(kb):
    """Regression: the safety question must return the actual precautions, not just a heading."""
    a = answer_question(kb.engine, "Which safety precautions apply before motor inspection?")
    assert a["sufficient"] and "lock out" in a["answer"].lower()
    lines = [ln for ln in a["answer"].splitlines() if ln.startswith("- ")]
    assert len(lines) == len(set(lines))
    assert "Document No" not in a["answer"]


def test_graph_edges_have_evidence(kb):
    e = kb.graph_edges()
    assert not e.empty and (e["evidence"].str.len() > 0).all() and (e["source_document"].str.len() > 0).all()
    assert {"has component", "experienced fault", "addressed by", "mentions", "reports"} <= set(e["relation"])


def test_path_safety():
    assert storage.safe_filename("../../etc/passwd") == "passwd"
    with pytest.raises(ValueError):
        _ORIG_INSIDE_PROJECT(Path("/etc/passwd"))


# ---------------- glass component ----------------
def test_glass_assets_and_fallback_present():
    for f in ("container.js", "glass.css", "html2canvas.min.js", "LICENSE"):
        assert (ASSET_DIR / f).exists()
    html = build_header_html("T", "S")
    assert "new Container(" in html and "CSS fallback" in html and "WebGL refraction" in html and "</script><script" not in html[:50]
