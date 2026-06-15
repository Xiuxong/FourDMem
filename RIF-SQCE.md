RIF Score & Quota Configuration Examples

# config/memory_v3.yaml
quota:
  L3: 300    # 核心画像 — 始终置顶
  L2: 600    # 场景知识 — 主力上下文
  L1: 450    # 原子事实 — 精确补充
  L0: 150    # 原始证据 — 仅在追问时展开
  overflow_policy: "downward_cascade"  # 未用完配额向下转移

rif_weights:
  recency: 0.3
  importance: 0.5
  frequency: 0.2

importance_presets:
  architecture_decision: 1.0
  user_explicit_command: 1.0
  bug_fix_rationale: 0.8
  tech_selection: 0.8
  code_style_preference: 0.5
  temporary_context: 0.2

temporal_gate:
  default_include_expired: false
  expired_label: "[EXPIRED]"

distillation:
  l1_to_l2_threshold: 5        # ≥5 条 L1 触发 L2 聚合
  l2_to_l3_promotion:
    min_references: 20
    min_span_days: 30
    require_approval: true
  force_provenance_chain: true  # 强制四维溯源链
  max_distillation_depth: 3
