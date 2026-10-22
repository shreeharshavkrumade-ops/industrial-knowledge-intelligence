"""Functional verification tests for Maintenance Log and Neural Search."""
import sys, os
sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path
import tempfile

# --- use an isolated test DB ---
import src.maintenance_log as ml_mod
_tmp = tempfile.mkdtemp()
ml_mod.ML_DB = Path(_tmp) / "test_ml.db"
ml_mod.init_db()

print("=== MAINTENANCE LOG TESTS ===")

# 1. Create a log
r = ml_mod.create_log({
    "equipment_id":   "P-101",
    "equipment_name": "Centrifugal Pump",
    "fault_title":    "Bearing overheating",
    "description":    "Pump bearing temperature exceeded 85°C during routine inspection.",
    "how_occurred":   "Observed during shift check at 14:00. Temperature rising steadily.",
    "observed_at":    "2026-10-22T08:00:00",
    "severity":       "High",
    "action_taken":   "Reduced load, applied cooling, notified maintenance.",
    "assigned_to":    "tech_ali",
    "status":         "In Progress",
}, username="smoketest_user")
assert r["ok"], f"Create failed: {r['error']}"
log_id = r["log_id"]
print(f"1. Create log: PASS ({log_id})")

# 2. Retrieve the log
rec = ml_mod.get_log(log_id)
assert rec["equipment_id"] == "P-101"
assert rec["severity"] == "High"
assert rec["status"] == "In Progress"
print("2. Get log: PASS")

# 3. Update status to Resolved
r2 = ml_mod.update_log(log_id, {
    "status":      "Resolved",
    "resolution":  "Replaced bearing assembly. Pump temperature normal after 2h test run.",
    "resolved_at": "2026-10-22T16:00:00",
}, username="tech_ali")
assert r2["ok"], f"Update failed: {r2['error']}"
rec2 = ml_mod.get_log(log_id)
assert rec2["status"] == "Resolved", f"Status not updated: {rec2['status']}"
assert "Replaced bearing" in rec2["resolution"]
print("3. Update to Resolved: PASS")

# 4. Audit trail records changes
audit = ml_mod.get_audit_trail(log_id)
assert not audit.empty, "Audit trail is empty"
assert "status" in audit["field_name"].values
assert "resolution" in audit["field_name"].values
# Verify resolved status was changed from In Progress -> Resolved
status_row = audit[audit["field_name"] == "status"].iloc[0]
assert status_row["old_value"] == "In Progress"
assert status_row["new_value"] == "Resolved"
assert status_row["changed_by"] == "tech_ali"
print(f"4. Audit trail: PASS ({len(audit)} change records)")

# 5. Unresolved fault not reported as resolved
r3 = ml_mod.create_log({
    "equipment_id": "C-202",
    "fault_title":  "Compressor vibration",
    "description":  "Excessive vibration on Compressor C-202.",
    "observed_at":  "2026-10-22T10:00:00",
    "severity":     "Critical",
    "status":       "Pending",
}, username="smoketest_user")
assert r3["ok"]
pending_rec = ml_mod.get_log(r3["log_id"])
assert pending_rec["status"] == "Pending"
assert not pending_rec["resolution"]
assert not pending_rec["resolved_at"]
print("5. Unresolved fault stays unresolved: PASS")

# 6. Validation rejects missing required fields
r4 = ml_mod.create_log({
    "equipment_id": "",
    "fault_title":  "Missing equip ID",
    "observed_at":  "2026-10-22T10:00:00",
    "severity":     "Low",
    "status":       "Pending",
}, username="smoketest_user")
assert not r4["ok"]
print("6. Validation rejects missing equipment ID: PASS")

# 7. Resolved status requires resolution text
r5 = ml_mod.create_log({
    "equipment_id": "V-301",
    "fault_title":  "Valve leak",
    "observed_at":  "2026-10-22T10:00:00",
    "severity":     "Medium",
    "status":       "Resolved",
    "resolution":   "",  # empty - should fail
}, username="smoketest_user")
assert not r5["ok"], "Should require resolution for Resolved status"
print("7. Resolved requires resolution text: PASS")

# 8. List logs - filter by equipment
logs_df = ml_mod.list_logs(equipment_id="P-101")
assert len(logs_df) >= 1
assert logs_df.iloc[0]["equipment_id"] == "P-101"
print(f"8. List + filter logs: PASS ({len(logs_df)} records for P-101)")

# 9. Text chunks for retrieval
chunks = ml_mod.logs_as_text_chunks()
assert len(chunks) >= 2
c0 = chunks[0]
assert "P-101" in c0["text"] or "C-202" in c0["text"]
assert "log_id" not in c0  # should be 'location'
assert c0["file_type"] == "log"
print(f"9. Text chunks for AI retrieval: PASS ({len(chunks)} chunks)")

# 10. DB persistence — re-init and re-read
ml_mod.init_db()  # safe to call again
rec3 = ml_mod.get_log(log_id)
assert rec3["status"] == "Resolved"
print("10. Persistence across re-init: PASS")

print("")
print("=== NEURAL SEARCH TEST ===")

# 11. Neural model loads and returns sensible similarity
from sentence_transformers import SentenceTransformer
import numpy as np
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
e1 = m.encode(["bearing failure overheating pump"], normalize_embeddings=True)
e2 = m.encode(["pump bearing temperature excessive heat"], normalize_embeddings=True)
e3 = m.encode(["compliance regulatory audit report"], normalize_embeddings=True)
sim_related   = float(np.dot(e1[0], e2[0]))
sim_unrelated = float(np.dot(e1[0], e3[0]))
assert sim_related > 0.70, f"Related similarity too low: {sim_related}"
assert sim_related > sim_unrelated, "Semantic ranking wrong"
print(f"11. Neural model similarity: PASS (related={sim_related:.3f}, unrelated={sim_unrelated:.3f})")

# 12. RetrievalEngine uses neural mode
import pandas as pd
from src.retrieval_engine import RetrievalEngine

test_chunks = pd.DataFrame([
    {"doc_id":"d1","document":"pump_manual.pdf","file_type":"pdf",
     "chunk_id":"d1-000","page":1,"location":"page 1",
     "text":"The centrifugal pump bearing must be lubricated every 500 hours. Overheating indicates lubrication failure.",
     "n_words":15},
    {"doc_id":"d2","document":"safety_guide.pdf","file_type":"pdf",
     "chunk_id":"d2-000","page":1,"location":"page 1",
     "text":"Electrical isolation must be verified before any maintenance work begins.",
     "n_words":12},
    {"doc_id":"d3","document":"compliance.pdf","file_type":"pdf",
     "chunk_id":"d3-000","page":1,"location":"page 1",
     "text":"Regulatory inspections must be scheduled annually per ISO 55001.",
     "n_words":10},
])
re = RetrievalEngine(test_chunks)
ok = re.load_semantic()
assert ok, f"Semantic load failed: {re.semantic_error}"
avail = re.availability()
assert avail["semantic"][0], "Semantic not available after load"
res = re.search("pump bearing overheating lubrication", mode="semantic", top_k=3)
assert not res.empty
assert res.iloc[0]["document"] == "pump_manual.pdf", f"Wrong top result: {res.iloc[0]['document']}"
print(f"12. Semantic search document retrieval: PASS (top: {res.iloc[0]['document']}, score={res.iloc[0]['score']:.3f})")

print("")
print("=== ALL TESTS PASSED ===")
