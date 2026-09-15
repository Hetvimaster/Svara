import hashlib
import json
import os
import time

class AuditLog:
    """Tamper-evident, append-only local log. Each entry's hash depends on
    every entry before it — altering or deleting a past entry breaks every
    hash that follows, which is detectable without a database or blockchain."""

    def __init__(self, log_path="data/audit_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            open(log_path, "w").close()

    def _last_hash(self):
        if os.path.getsize(self.log_path) == 0:
            return "0" * 64  # genesis hash — first entry chains to this
        with open(self.log_path) as f:
            lines = [line for line in f if line.strip()]
        if not lines:
            return "0" * 64
        return json.loads(lines[-1])["entry_hash"]

    def append(self, verdict: dict):
        """verdict: e.g. {"spoof_score":..., "risk_score":..., "tier":...}
        Never store raw audio or embeddings here — only the scored verdict,
        per the architecture's privacy design (audio is deleted post-scoring)."""
        prev_hash = self._last_hash()
        entry = {
            "timestamp": time.time(),
            "verdict": verdict,
            "prev_hash": prev_hash,
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["entry_hash"] = entry_hash

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry_hash

    def verify_chain(self):
        """Walk the whole log, recomputing each hash. Returns (True, None)
        if intact, or (False, line_number) at the first broken link."""
        prev_hash = "0" * 64
        with open(self.log_path) as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                stored_hash = entry.pop("entry_hash")
                if entry["prev_hash"] != prev_hash:
                    return False, i
                recomputed = hashlib.sha256(
                    json.dumps(entry, sort_keys=True).encode()
                ).hexdigest()
                if recomputed != stored_hash:
                    return False, i
                prev_hash = stored_hash
        return True, None


if __name__ == "__main__":
    log = AuditLog(log_path="../data/audit_log_test.jsonl")
    log.append({"spoof_score": 0.99, "risk_score": 72.26, "tier": "medium"})
    log.append({"spoof_score": 0.14, "risk_score": 5.51, "tier": "low"})
    ok, bad_line = log.verify_chain()
    print(f"Chain intact: {ok}" if ok else f"Chain broken at line {bad_line}")