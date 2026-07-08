"""
nexroute_router.py
==================
NexRoute 듀얼-축 라우터 (Domain Axis R_D + Function Axis R_F)

논문: "NexRoute: A Dual-Axis Nexus Routing Framework for Adaptive Expert
      Selection in Cross-Domain Large Language Models"

아키텍처:
  - 공유 인코더: DeBERTa-large  (h ∈ R^1024)
  - R_D: 4-class binary multi-label sigmoid  →  CROSS 판별 (τ/δ 임계값)
  - R_F: 4-class softmax  →  기능 카테고리
  - Joint loss: L = λ·L_D + (1−λ)·L_F

환경: PyTorch ≥ 2.0, transformers ≥ 4.40, VRAM ≥ 8 GB (추론) / 14.8 GB (학습)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


# ──────────────────────────────────────────────────────────────────────────────
# 1. 상수 및 기본값
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_LABELS: list[str]   = ["LAW", "MED", "SOC", "STEM"]
FUNCTION_LABELS: list[str] = ["What", "How", "Why", "Compare"]
CROSS_LABEL: str           = "CROSS"

NUM_DOMAINS:    int = len(DOMAIN_LABELS)    # 4
NUM_FUNCTIONS:  int = len(FUNCTION_LABELS)  # 4

DEFAULT_TAU:    float = 0.30   # τ — CROSS 활성화: secondary score ≥ τ
DEFAULT_DELTA:  float = 0.40   # δ — CROSS 활성화: s1 − s2 ≤ δ
DEFAULT_LAMBDA: float = 0.50   # λ — joint loss 도메인 가중치


# ──────────────────────────────────────────────────────────────────────────────
# 2. 출력 데이터클래스
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RouterOutput:
    """단일 쿼리에 대한 추론 결과.

    Attributes:
        domain: 예측 도메인 레이블 ("LAW"|"MED"|"SOC"|"STEM"|"CROSS")
        function: 예측 기능 레이블 ("What"|"How"|"Why"|"Compare")
        domain_scores: 4개 기본 도메인 sigmoid 점수  shape (4,)
        function_probs: 4개 기능 softmax 확률       shape (4,)
        is_cross: CROSS 판별 결과
        top2_domains: (primary, secondary) 도메인 이름 쌍
        top2_scores: (s1, s2) — 상위 2개 sigmoid 점수
        nexus_point: (domain, function) 튜플 — LoRA 전문가 선택 키
    """
    domain:         str
    function:       str
    domain_scores:  torch.Tensor
    function_probs: torch.Tensor
    is_cross:       bool
    top2_domains:   tuple[str, str]
    top2_scores:    tuple[float, float]
    nexus_point:    tuple[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.nexus_point = (self.domain, self.function)

    def __repr__(self) -> str:
        s1, s2 = self.top2_scores
        return (
            f"RouterOutput(\n"
            f"  nexus_point = {self.nexus_point},\n"
            f"  is_cross    = {self.is_cross},\n"
            f"  top2        = {self.top2_domains[0]}({s1:.3f}) / "
            f"{self.top2_domains[1]}({s2:.3f}),\n"
            f"  fn_probs    = {{ {', '.join(f'{l}:{p:.3f}' for l, p in zip(FUNCTION_LABELS, self.function_probs.tolist()))} }}\n"
            f")"
        )


@dataclass
class RouterLossOutput:
    """훈련 스텝 출력.

    Attributes:
        loss: joint loss (역전파 대상)
        loss_domain: R_D binary cross-entropy
        loss_function: R_F categorical cross-entropy
        domain_logits: R_D 원시 출력  shape (B, 4)
        function_logits: R_F 원시 출력 shape (B, 4)
    """
    loss:            torch.Tensor
    loss_domain:     torch.Tensor
    loss_function:   torch.Tensor
    domain_logits:   torch.Tensor
    function_logits: torch.Tensor


# ──────────────────────────────────────────────────────────────────────────────
# 3. 듀얼-축 라우터 모듈
# ──────────────────────────────────────────────────────────────────────────────

class DualAxisRouter(nn.Module):
    """NexRoute 듀얼-축 라우터.

    DeBERTa-large 공유 인코더 위에 도메인 헤드 R_D와 기능 헤드 R_F를
    병렬로 연결한다. 두 헤드는 동일한 [CLS] 표현 h ∈ R^1024를 입력으로
    받아 독립적으로 분류를 수행하며, 학습 시에는 Joint loss로 공동 최적화된다.

    CROSS 판별 (추론 전용):
        isCross(q) = True  iff  s2 ≥ τ  AND  s1 − s2 ≤ δ
        τ=0.30, δ=0.40 (논문 기본값)

    Args:
        encoder_name: HuggingFace 모델 ID
        tau: CROSS 활성화 최소 secondary score
        delta: top-2 gap 허용 최대치
        lambda_weight: joint loss 도메인 가중치 λ
        dropout: classifier head dropout 비율
        freeze_encoder: True면 인코더 파라미터 동결 (VRAM 절약)
    """

    def __init__(
        self,
        encoder_name: str   = "microsoft/deberta-large",
        tau:          float = DEFAULT_TAU,
        delta:        float = DEFAULT_DELTA,
        lambda_weight:float = DEFAULT_LAMBDA,
        dropout:      float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        # ── 공유 인코더 ──────────────────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size: int = self.encoder.config.hidden_size  # DeBERTa-large: 1024

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        # ── 분류 헤드 ─────────────────────────────────────────────────────────
        # R_D: binary multi-label — sigmoid + BCE
        self.domain_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_DOMAINS),    # 1024 → 4
        )
        # R_F: single-label — softmax + CE
        self.function_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_FUNCTIONS),  # 1024 → 4
        )

        # ── 임계값 및 하이퍼파라미터 (학습 파라미터 아님) ─────────────────────
        self.tau:           float = tau
        self.delta:         float = delta
        self.lambda_weight: float = lambda_weight

    # ── 인코딩 ────────────────────────────────────────────────────────────────

    def _encode(
        self,
        input_ids:      torch.Tensor,   # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """DeBERTa-large [CLS] 벡터 추출.

        Args:
            input_ids: 토큰 인덱스  (B, L)
            attention_mask: 패딩 마스크 (B, L)

        Returns:
            h: [CLS] 표현  (B, H), H=1024
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # [CLS] 위치

    # ── CROSS 판별 (정적) ─────────────────────────────────────────────────────

    @staticmethod
    def _is_cross(s1: float, s2: float, tau: float, delta: float) -> bool:
        """CROSS 도메인 판별 함수.

        논문 Eq. (isCross): s2 ≥ τ  AND  s1 − s2 ≤ δ

        Args:
            s1: 1위 도메인 sigmoid 점수
            s2: 2위 도메인 sigmoid 점수
            tau: 최소 활성화 임계값
            delta: 최대 gap 허용치

        Returns:
            True면 CROSS 도메인으로 판별
        """
        return (s2 >= tau) and ((s1 - s2) <= delta)

    # ── 포워드 패스 ───────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:       torch.Tensor,
        attention_mask:  torch.Tensor,
        domain_labels:   Optional[torch.Tensor] = None,  # (B, 4) float — 훈련용
        function_labels: Optional[torch.Tensor] = None,  # (B,)   long  — 훈련용
    ) -> RouterOutput | RouterLossOutput:
        """포워드 패스 (훈련/추론 분기).

        Args:
            input_ids: 토크나이저 출력 (B, L)
            attention_mask: 패딩 마스크 (B, L)
            domain_labels: 다중 레이블 one-hot (B, 4) — None이면 추론 모드
            function_labels: 단일 레이블 인덱스 (B,) — None이면 추론 모드

        Returns:
            훈련 시: RouterLossOutput
            추론 시 (B=1): RouterOutput
        """
        h = self._encode(input_ids, attention_mask)  # (B, 1024)

        domain_logits   = self.domain_head(h)    # (B, 4)
        function_logits = self.function_head(h)  # (B, 4)

        if domain_labels is not None and function_labels is not None:
            return self._compute_loss(
                domain_logits, function_logits, domain_labels, function_labels
            )

        # 추론: 배치 첫 번째 샘플만 디코딩 (단일 쿼리 전제)
        return self._decode(domain_logits[0], function_logits[0])

    # ── 손실 계산 (훈련) ──────────────────────────────────────────────────────

    def _compute_loss(
        self,
        domain_logits:   torch.Tensor,  # (B, 4)
        function_logits: torch.Tensor,  # (B, 4)
        domain_labels:   torch.Tensor,  # (B, 4) float  multi-label one-hot
        function_labels: torch.Tensor,  # (B,)   long   single-label index
    ) -> RouterLossOutput:
        """Joint loss 계산.

        L_D = BCE(sigmoid(logits_D), y_D)   — 다중 레이블 도메인
        L_F = CE(logits_F, y_F)             — 단일 레이블 기능
        L   = λ·L_D + (1−λ)·L_F

        Args:
            domain_logits: R_D 출력 (B, 4)
            function_logits: R_F 출력 (B, 4)
            domain_labels: 도메인 다중 레이블 (B, 4)
            function_labels: 기능 단일 레이블 (B,)

        Returns:
            RouterLossOutput
        """
        loss_d = F.binary_cross_entropy_with_logits(
            domain_logits,
            domain_labels.float(),
        )
        loss_f = F.cross_entropy(function_logits, function_labels)

        # CROSS 샘플은 두 도메인 모두 1인 multi-hot이므로 L_D가 자연스럽게 처리
        joint = self.lambda_weight * loss_d + (1.0 - self.lambda_weight) * loss_f

        return RouterLossOutput(
            loss=joint,
            loss_domain=loss_d,
            loss_function=loss_f,
            domain_logits=domain_logits,
            function_logits=function_logits,
        )

    # ── 단일 샘플 디코딩 (추론) ───────────────────────────────────────────────

    def _decode(
        self,
        domain_logit:   torch.Tensor,  # (4,)
        function_logit: torch.Tensor,  # (4,)
    ) -> RouterOutput:
        """로짓 → RouterOutput 변환.

        Args:
            domain_logit: R_D 단일 샘플 로짓 (4,)
            function_logit: R_F 단일 샘플 로짓 (4,)

        Returns:
            RouterOutput
        """
        domain_scores  = torch.sigmoid(domain_logit)          # (4,)
        function_probs = F.softmax(function_logit, dim=-1)     # (4,)

        # top-2 도메인 점수 및 인덱스
        top2_vals, top2_idx = torch.topk(domain_scores, k=2)
        s1, s2 = top2_vals[0].item(), top2_vals[1].item()
        primary_idx   = top2_idx[0].item()
        secondary_idx = top2_idx[1].item()

        # CROSS 판별
        is_cross     = self._is_cross(s1, s2, self.tau, self.delta)
        domain_label = CROSS_LABEL if is_cross else DOMAIN_LABELS[primary_idx]

        function_label = FUNCTION_LABELS[function_probs.argmax().item()]

        return RouterOutput(
            domain        = domain_label,
            function      = function_label,
            domain_scores = domain_scores.detach().cpu(),
            function_probs= function_probs.detach().cpu(),
            is_cross      = is_cross,
            top2_domains  = (DOMAIN_LABELS[primary_idx], DOMAIN_LABELS[secondary_idx]),
            top2_scores   = (s1, s2),
        )

    # ── 배치 추론 (VRAM-safe) ─────────────────────────────────────────────────

    @torch.no_grad()
    def batch_predict(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size:     int = 16,
    ) -> list[RouterOutput]:
        """VRAM 안전 배치 추론.

        14.8 GB 환경에서 대량 쿼리를 처리할 때 chunk_size로 분할,
        각 청크 처리 후 중간 텐서를 즉시 해제한다.

        Args:
            input_ids: (N, L)
            attention_mask: (N, L)
            chunk_size: 청크당 샘플 수 (GPU 메모리에 따라 조정)

        Returns:
            results: 길이 N의 RouterOutput 리스트
        """
        self.eval()
        results: list[RouterOutput] = []
        n = input_ids.size(0)

        for start in range(0, n, chunk_size):
            end  = min(start + chunk_size, n)
            c_ids  = input_ids[start:end]
            c_mask = attention_mask[start:end]

            h        = self._encode(c_ids, c_mask)
            d_logits = self.domain_head(h)
            f_logits = self.function_head(h)

            for i in range(d_logits.size(0)):
                results.append(self._decode(d_logits[i], f_logits[i]))

            # 청크 완료 후 캐시 해제 (OOM 방지)
            del h, d_logits, f_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. 토크나이저 래퍼 (편의 유틸)
# ──────────────────────────────────────────────────────────────────────────────

class RouterTokenizer:
    """DeBERTa-large 토크나이저 래퍼.

    Args:
        tokenizer_name: HuggingFace 토크나이저 이름
        max_length: 최대 시퀀스 길이
        device: 텐서 디바이스
    """

    def __init__(
        self,
        tokenizer_name: str  = "microsoft/deberta-large",
        max_length:     int  = 256,
        device:         str  = "cpu",
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.max_length = max_length
        self.device     = torch.device(device)

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """텍스트 리스트를 모델 입력으로 변환.

        Args:
            texts: 쿼리 문자열 리스트

        Returns:
            {"input_ids": Tensor, "attention_mask": Tensor}  (device로 이동)
        """
        enc = self.tokenizer(
            texts,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()
                if k in ("input_ids", "attention_mask")}


# ──────────────────────────────────────────────────────────────────────────────
# 5. 더미 데이터 실행 예시
# ──────────────────────────────────────────────────────────────────────────────

def _make_dummy_batch(
    batch_size: int,
    seq_len:    int,
    vocab_size: int,
    device:     torch.device,
) -> dict[str, torch.Tensor]:
    """실제 토크나이저 없이 임의 정수 텐서로 배치 생성."""
    return {
        "input_ids":      torch.randint(1, vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long,  device=device),
    }


def demo_inference(router: DualAxisRouter, device: torch.device) -> None:
    """단일 쿼리 추론 데모."""
    print("\n" + "═" * 60)
    print("  [추론 모드] 단일 쿼리")
    print("═" * 60)

    router.eval()
    batch = _make_dummy_batch(1, seq_len=64, vocab_size=50265, device=device)

    with torch.no_grad():
        result: RouterOutput = router(**batch)

    print(result)
    print(f"  → 선택 LoRA 전문가: e_{result.domain}  |  scaffold: {result.function}\n")


def demo_training_step(router: DualAxisRouter, device: torch.device) -> None:
    """미니 배치 학습 스텝 데모."""
    print("═" * 60)
    print("  [훈련 모드] 미니배치 (B=4)")
    print("═" * 60)

    router.train()
    B = 4

    batch = _make_dummy_batch(B, seq_len=64, vocab_size=50265, device=device)

    # 도메인 다중 레이블 (예: 샘플 0은 CROSS=LAW+MED, 나머지는 단일 도메인)
    # shape (B, 4) — LAW, MED, SOC, STEM 순
    domain_labels = torch.zeros(B, NUM_DOMAINS, device=device)
    domain_labels[0, 0] = 1.0  # LAW
    domain_labels[0, 1] = 1.0  # MED  → CROSS (LAW+MED)
    domain_labels[1, 2] = 1.0  # SOC
    domain_labels[2, 3] = 1.0  # STEM
    domain_labels[3, 0] = 1.0  # LAW

    # 기능 단일 레이블 (0=What, 1=How, 2=Why, 3=Compare)
    function_labels = torch.tensor([2, 0, 1, 3], dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, router.parameters()),
        lr=2e-5,
    )
    optimizer.zero_grad()
    out: RouterLossOutput = router(**batch,
                                   domain_labels=domain_labels,
                                   function_labels=function_labels)
    out.loss.backward()
    optimizer.step()

    print(f"  joint loss   : {out.loss.item():.4f}")
    print(f"  loss_domain  : {out.loss_domain.item():.4f}")
    print(f"  loss_function: {out.loss_function.item():.4f}\n")


def demo_batch_predict(router: DualAxisRouter, device: torch.device) -> None:
    """배치 추론 (VRAM-safe chunk) 데모."""
    print("═" * 60)
    print("  [배치 추론] N=6, chunk_size=2")
    print("═" * 60)

    N = 6
    batch = _make_dummy_batch(N, seq_len=64, vocab_size=50265, device=device)
    results = router.batch_predict(**batch, chunk_size=2)

    for i, r in enumerate(results):
        s1, s2 = r.top2_scores
        print(f"  Q{i}: nexus={r.nexus_point}  "
              f"top2=[{r.top2_domains[0]}:{s1:.3f}, {r.top2_domains[1]}:{s2:.3f}]  "
              f"CROSS={r.is_cross}")
    print()


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """더미 데이터로 전체 파이프라인을 검증한다.

    실제 DeBERTa-large 체크포인트 없이도 동작하도록
    config-only 로드(AutoModel.from_config)를 사용한다.
    """
    import os
    from transformers import AutoConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n디바이스: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}  ({props.total_memory / 1e9:.1f} GB)")

    # ── 더미 모델 생성 (체크포인트 없이 랜덤 가중치) ─────────────────────────
    print("\n[1/3] 더미 DualAxisRouter 초기화 (랜덤 가중치)...")
    # 빠른 테스트를 위해 hidden_size를 축소한 tiny config 사용
    cfg = AutoConfig.from_pretrained("microsoft/deberta-large")
    cfg.hidden_size          = 128   # ← 테스트용 축소 (실제: 1024)
    cfg.intermediate_size    = 512
    cfg.num_hidden_layers    = 2
    cfg.num_attention_heads  = 4
    cfg.update({"pooler_hidden_size": 128})

    # config만으로 초기화 (허깅페이스 다운로드 불필요)
    from transformers import DebertaModel

    class _TinyRouter(DualAxisRouter):
        """테스트용 tiny 인코더를 주입한 서브클래스."""
        def __init__(self) -> None:
            # nn.Module.__init__만 호출하고 수동으로 구성
            nn.Module.__init__(self)
            self.encoder = DebertaModel(cfg)
            hidden_size  = cfg.hidden_size  # 128
            self.domain_head   = nn.Sequential(nn.Dropout(0.1),
                                               nn.Linear(hidden_size, NUM_DOMAINS))
            self.function_head = nn.Sequential(nn.Dropout(0.1),
                                               nn.Linear(hidden_size, NUM_FUNCTIONS))
            self.tau           = DEFAULT_TAU
            self.delta         = DEFAULT_DELTA
            self.lambda_weight = DEFAULT_LAMBDA

    router = _TinyRouter().to(device)
    total_params = sum(p.numel() for p in router.parameters())
    trainable    = sum(p.numel() for p in router.parameters() if p.requires_grad)
    print(f"   전체 파라미터: {total_params:,}  |  학습 가능: {trainable:,}")

    # ── 데모 실행 ────────────────────────────────────────────────────────────
    print("\n[2/3] 추론 + 훈련 스텝 데모")
    demo_inference(router, device)
    demo_training_step(router, device)

    print("[3/3] 배치 추론 (chunk) 데모")
    demo_batch_predict(router, device)

    # ── VRAM 사용량 리포트 ───────────────────────────────────────────────────
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved  = torch.cuda.memory_reserved(device)  / 1e9
        print(f"GPU 메모리: allocated={allocated:.2f} GB / reserved={reserved:.2f} GB")

    print("✓ 모든 데모 완료\n")


if __name__ == "__main__":
    main()
