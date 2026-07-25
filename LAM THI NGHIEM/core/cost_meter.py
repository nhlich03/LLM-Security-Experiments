"""
cost_meter.py — bộ đo cost dùng chung cho mọi phương pháp phòng thủ.

Đo ở đơn vị THÔ (tiền quy đổi tính sau):
  - API call  -> token (prompt_tokens + completion_tokens) từ field `usage`
  - Local infer -> giây (perf_counter + torch.cuda.synchronize)
  - Train      -> giây (một lần)

Tách theo call để biết cost đến từ đâu (target / phụ trợ / decode).
Xuất: file chi tiết (mỗi call 1 dòng) + tóm tắt (mean/std mỗi request).

Nguyên tắc công bằng: mọi phương pháp đo trên CÙNG GPU, CÙNG max_tokens,
cùng điều kiện sinh. Giá tiền để placeholder, nhân sau.
"""

import os
import time
import json
from contextlib import contextmanager

import pandas as pd

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    _HAS_CUDA = False

# --- Placeholder đơn giá (điền sau, KHÔNG dùng lúc đo) ---
PRICE_GPU_PER_SEC     = None   # $/giây thuê GPU (điền khi biết loại GPU)
PRICE_API_PER_1K_IN   = None   # $/1K input token
PRICE_API_PER_1K_OUT  = None   # $/1K output token


class CostMeter:
    def __init__(self, method_name, defense_type):
        """
        method_name: tên phương pháp (vd "SAGE", "erase_and_check")
        defense_type: "pre" | "post" | "in" | "intra"
        """
        self.method = method_name
        self.type = defense_type
        self._calls = []            # mỗi call: 1 dòng chi tiết
        self._cur_request = None    # request_id đang mở
        self.train_sec = 0.0        # cost train (một lần)

    # ---------------- REQUEST: gom các call của 1 lượt input->output ----------------
    @contextmanager
    def request(self, request_id):
        """Bọc quanh toàn bộ xử lý 1 prompt. Mọi call bên trong gắn vào request này."""
        self._cur_request = request_id
        try:
            yield self
        finally:
            self._cur_request = None

    # ---------------- API call: đo token ----------------
    def record_api(self, label, response):
        """
        Gọi ngay sau một API call. Tự lấy token từ response.usage.
        label: vai trò của call này, vd "target", "prompt_filter", "guard", "judge".
        response: object trả về từ client.chat.completions.create(...)
        """
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        self._calls.append({
            "method": self.method, "type": self.type,
            "request_id": self._cur_request, "call_label": label,
            "kind": "api", "api_in_tokens": in_tok, "api_out_tokens": out_tok,
            "local_sec": 0.0,
        })
        return in_tok, out_tok

    # ---------------- Local infer: đo giây ----------------
    @contextmanager
    def local(self, label):
        """
        Bọc quanh 1 đoạn infer local (vd model.generate hoặc 1 forward pass).
        Đo chính xác GPU work bằng synchronize trước/sau.
        label: "target", "expert_decode", "probe", "first_token_forward"...
        """
        if _HAS_CUDA:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if _HAS_CUDA:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self._calls.append({
                "method": self.method, "type": self.type,
                "request_id": self._cur_request, "call_label": label,
                "kind": "local", "api_in_tokens": 0, "api_out_tokens": 0,
                "local_sec": dt,
            })

    # ---------------- Train: một lần ----------------
    @contextmanager
    def train(self):
        """Bọc quanh toàn bộ quá trình train (chỉ intra, hoặc train model phụ)."""
        if _HAS_CUDA:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if _HAS_CUDA:
                torch.cuda.synchronize()
            self.train_sec += time.perf_counter() - t0

    # ---------------- Xuất kết quả ----------------
    def to_frames(self):
        """Trả về (df_chi_tiet, df_tom_tat_per_request)."""
        detail = pd.DataFrame(self._calls)
        if len(detail) == 0:
            return detail, pd.DataFrame()
        # gom mỗi request: cộng token + giây, đếm số call
        per_req = detail.groupby("request_id").agg(
            n_calls=("call_label", "size"),
            api_in_tokens=("api_in_tokens", "sum"),
            api_out_tokens=("api_out_tokens", "sum"),
            local_sec=("local_sec", "sum"),
        ).reset_index()
        return detail, per_req

    def save(self, detail_path, summary_path):
        detail, _ = self.to_frames()
        # MERGE with existing detail so cost accumulates across resume/--limit runs
        # (this run only recorded the newly-generated rows). Dedup by request+call.
        if os.path.exists(detail_path):
            try:
                old = pd.read_csv(detail_path)
                detail = (pd.concat([old, detail], ignore_index=True)
                          if len(detail) else old)
                detail = detail.drop_duplicates(subset=["request_id", "call_label"], keep="last")
            except Exception:
                pass
        if len(detail) == 0:
            print(f"\n=== COST [{self.method}] (type={self.type}) ===")
            print("Khong co call nao -> khong ghi.")
            return
        # recompute per-request summary from the merged detail
        per_req = detail.groupby("request_id").agg(
            n_calls=("call_label", "size"),
            api_in_tokens=("api_in_tokens", "sum"),
            api_out_tokens=("api_out_tokens", "sum"),
            local_sec=("local_sec", "sum"),
        ).reset_index()
        # train cost = MOT LAN (khong theo request) -> broadcast de compare_methods doc duoc;
        # neu resume ma khong train lai (train_sec=0) thi giu gia tri cu trong file.
        train_sec = self.train_sec
        if train_sec == 0 and os.path.exists(summary_path):
            try:
                old_ts = pd.read_csv(summary_path).get("train_sec")
                if old_ts is not None and len(old_ts):
                    train_sec = float(old_ts.max())
            except Exception:
                pass
        per_req["train_sec"] = train_sec
        detail.to_csv(detail_path, index=False)
        per_req.to_csv(summary_path, index=False)

        # tóm tắt in ra màn hình
        print(f"\n=== COST [{self.method}] (type={self.type}) ===")
        if len(per_req):
            n = len(per_req)
            print(f"Số request: {n}")
            print(f"Infer / request  -> token in:  {per_req.api_in_tokens.mean():.0f} ± {per_req.api_in_tokens.std():.0f}")
            print(f"                    token out: {per_req.api_out_tokens.mean():.0f} ± {per_req.api_out_tokens.std():.0f}")
            print(f"                    local sec: {per_req.local_sec.mean():.3f} ± {per_req.local_sec.std():.3f}")
            print(f"                    số call:   {per_req.n_calls.mean():.1f}")
        print(f"Train (một lần):     {self.train_sec:.1f} giây" if self.train_sec else "Train: không có")
        print(f"Đã lưu: {detail_path} | {summary_path}")


# =====================================================================
# VÍ DỤ NHÚNG THEO TỪNG LOẠI (copy đúng mẫu của phương pháp bạn)
# =====================================================================
if __name__ == "__main__":

    # ---- PRE, single-call (SAGE): 1 call target, prompt đã ghép instruction ----
    # meter = CostMeter("SAGE", "pre")
    # for row in data:
    #     with meter.request(row.id):
    #         resp = client.chat.completions.create(model=TARGET, messages=[...])  # API
    #         meter.record_api("target", resp)
    #     # nếu target local: with meter.local("target"): out = model.generate(...)

    # ---- PRE, multi-call (G4D): call phụ phân tích intent + call target ----
    # meter = CostMeter("G4D", "pre")
    # with meter.request(row.id):
    #     r1 = client.chat.completions.create(...);  meter.record_api("intent_guide", r1)
    #     r2 = client.chat.completions.create(...);  meter.record_api("target", r2)

    # ---- PRE, O(n) (erase-and-check): gọi safety filter nhiều lần ----
    # meter = CostMeter("erase_and_check", "pre")
    # with meter.request(row.id):
    #     for sub in erased_subsequences(prompt):     # O(n) lần
    #         with meter.local("filter_check"): flag = safety_filter(sub)
    #     with meter.local("target"): out = model.generate(prompt)   # nếu pass

    # ---- POST (Self-Refine): target + feedback + refine (nhiều vòng) ----
    # meter = CostMeter("self_refine", "post")
    # with meter.request(row.id):
    #     r = client.chat.completions.create(...); meter.record_api("target", r)
    #     for k in range(rounds):
    #         rf = client.chat.completions.create(...); meter.record_api("feedback", rf)
    #         rr = client.chat.completions.create(...); meter.record_api("refine", rr)

    # ---- IN, decoding-time (SafeDecoding/GeDi): đo cả vòng generate có can thiệp ----
    # meter = CostMeter("safedecoding", "in")
    # with meter.request(row.id):
    #     with meter.local("guided_decode"): out = safe_generate(model, expert, prompt)
    #   (overhead = so với dòng no_defense đo cùng cách, tính lúc gộp bảng)

    # ---- IN/PRE, first-token (FJD): 1 forward token để flag, rồi mới sinh nếu pass ----
    # meter = CostMeter("FJD", "pre")
    # with meter.request(row.id):
    #     with meter.local("first_token_forward"): flag = fjd_check(model, prompt)
    #     if not flag:
    #         with meter.local("target"): out = model.generate(prompt)

    # ---- INTRA (SecAlign): train một lần + infer như thường ----
    # meter = CostMeter("secalign", "intra")
    # with meter.train():
    #     dpo_train(model, data)          # cost train (một lần)
    # for row in data:
    #     with meter.request(row.id):
    #         with meter.local("target"): out = tuned_model.generate(row.prompt)

    # meter.save("cost_detail.csv", "cost_summary.csv")
    pass
