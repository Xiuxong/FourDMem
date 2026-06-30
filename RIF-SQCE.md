# RIF-U Score, Quota & Cognitive Evolution Configuration

> **说明**: 本文档定义了 FourDMem 的核心运行参数规范。系统采用 **RIF-U (Recency, Importance, Frequency, Utility)** 四维评分，时间计算基于**主观时间 (Active Ticks)**，并包含**元认知路由**与**认知进化引擎**的控制参数。
>
> ⚠️ 注意：以下是参数规范，实际值硬编码在 Rust/Python 源码中。

```yaml
# FourDMem 参数规范 (非实际配置文件)

# ==========================================
# 1. Token 预算与层级配额 (Token Budget & Quota)
# ==========================================
quota:
  total_budget: 1500
  # 注: L4 (Meta-Cognition/Observer) 作为系统级规则注入，不占用动态检索配额
  L3: 300    # 20% 核心画像 — 始终置顶，全局共享
  L2: 600    # 40% 场景知识 — 主力上下文，条件化规则
  L1: 450    # 30% 原子事实 — 精确补充，图谱节点
  L0: 150    # 10% 原始证据 — 仅在追问/自动下钻时展开
  overflow_policy: "downward_cascade"  # 未用完的配额向下层转移 (L3->L2->L1->L0)

# ==========================================
# 2. RIF-U 评分权重 (Recency, Importance, Frequency, Utility)
# ==========================================
rif_weights:
  recency: 0.25      # 时效性 (强制基于 Active Ticks 主观时间计算)
  importance: 0.35   # 重要性 (基于预设标签或 LLM 评估)
  frequency: 0.15    # 频率 (被检索、引用和证实的次数)
  utility: 0.25      # [V4.0] 效用 (基于 submit_feedback 的外部 Reward/Punishment)

# ==========================================
# 3. 重要性与效用预设 (Importance & Utility Presets)
# ==========================================
importance_presets:
  architecture_decision: 1.0
  user_explicit_command: 1.0
  bug_fix_rationale: 0.9
  tech_selection: 0.8
  code_style_preference: 0.6
  temporary_context: 0.2

utility_presets:
  # 初始效用值，后续由 submit_feedback 动态修改
  verified_fact: 0.5
  unverified_hypothesis: 0.0
  deprecated_pattern: -0.8
  critical_pitfall: 1.0  # 痛点记忆，强制高 Utility 且享有衰减免疫

# ==========================================
# 4. 主观时间与智能时序门控 (Subjective Time & Temporal Gate)
# ==========================================
temporal_gate:
  default_include_expired: false
  expired_label: "[EXPIRED]"
  
  # [V4.0] 长休眠宽容期 (Dormancy Grace Period)
  # 当物理停机 > 7天，不直接丢弃过期版本，而是标记为历史上下文供断层反思
  physical_dormancy_grace_days: 7
  historical_context_label: "[HISTORICAL_CONTEXT]"
  
  # 保质期分类 (Shelf-life Categories)
  shelf_life:
    physical: "30_days"      # 物理时间衰减 (如: 临时 API 密钥、短期活动)
    subjective: "90_ticks"   # [V4.0] 主观时间衰减 (如: 业务逻辑、代码规范，基于交互次数)
    immune: "never"          # 免疫衰减 (如: L3 核心画像、高危痛点记忆)

# ==========================================
# 5. 元认知与渐进式披露 (Metacognition & Progressive Disclosure)
# ==========================================
metacognition:
  # 置信度评估：当 L2 摘要的语义置信度低于此阈值时，自动触发下钻
  confidence_drill_down_threshold: 0.65
  
  # 异步预取：返回 L2 时，后台预取关联度最高的 N 个 L1 节点放入缓存
  prefetch_top_k: 2
  
  # [V4.0] 观察者干预 (Observer Intervention)
  # 当连续 N 次检索导致 Agent 执行失败（同化失败），允许 L4 观察者接管并修改权重
  observer_takeover_after_failures: 3

# ==========================================
# 6. 生命周期与蒸馏 (Lifecycle & Distillation)
# ==========================================
distillation:
  l1_to_l2_threshold: 5        # ≥5 条相关 L1 触发 L2 场景聚合
  
  l2_to_l3_promotion:
    min_references: 20
    # [V4.0] 将物理时间跨度改为主观交互刻度 (Active Ticks)
    min_span_ticks: 50         
    require_approval: true     # 晋升 L3 需人类或高级 Agent 审批
    
  force_provenance_chain: true  # 强制 L3→L2→L1→L0 四维溯源链
  max_distillation_depth: 3

# [V4.0] 梦境修剪 (Sleep & Prune)
dream_pruning:
  # 触发条件：每累积 100 个 Active Ticks 或每天凌晨 3 点
  trigger_interval_ticks: 100
  cron_fallback: "0 3 * * *"
  
  # 衰减曲线：艾宾浩斯遗忘曲线 (基于 Active Ticks)
  decay_function: "ebbinghaus"
  
  # 免疫名单：以下类型的记忆不参与衰减
  immunity_tags: ["critical_pitfall", "l3_core", "high_utility"]

# ==========================================
# 7. 认知进化与怪圈引擎 (Cognitive Evolution Engine) [V4.0 核心]
# ==========================================
evolution:
  # [生物学] 认知髓鞘化 (思维宏编译)
  myelination:
    # 当某条 L1->L1->L2 推理路径被成功调用超过 N 次，编译为直觉宏
    macro_compilation_threshold: 10
    macro_success_rate_required: 0.90

  # [物理学] 拓扑相变 (跨域顿悟)
  topological_phase_transition:
    # 使用持久同调 (Persistent Homology) 计算 Betti 数
    enable_tda_monitor: true
    # 当图谱的一维环 (Betti-1) 数量达到此阈值，触发 GNN 跨域类比搜索
    betti_1_critical_threshold: 45

  # [哲学] 范式转移 (认知危机与重构)
  paradigm_shift:
    # 同化失败率 (检索出的记忆被采纳后导致报错的比例) 监控窗口
    failure_rate_window_ticks: 20
    # 失败率超过此阈值，冻结该领域写入，触发辩证反思 Agent
    crisis_trigger_failure_rate: 0.30   # 代码实际值: 0.30

  # [复杂系统] 认知基因组与沙盒 (Cognitive Genome & Sandbox)
  genome_sandbox:
    # 允许 LLM 自动生成 Python 检索插件 (Auto-Plugin)
    enable_auto_plugin_generation: true
    # 沙盒执行超时与内存限制
    sandbox_timeout_ms: 5000
    sandbox_memory_limit_mb: 256
    # 变异后的 DNA (检索权重/Prompt) 必须在沙盒中超过此适应度分数才能热替换
    min_fitness_score_for_hotswap: 0.85
```

## 源码位置

| 参数 | 源文件 |
|:---|:---|
| RIF-U 权重 | `crates/retrieval-core/src/rif_u.rs` |
| Token 配额 | `crates/retrieval-core/src/token_budget.rs` |
| 置信度阈值 | `crates/retrieval-core/src/router.rs` |
| 梦境修剪 | `python/mcp_server/lifecycle.py` |
| 髓鞘化 | `python/evolution/myelination.py` |
| 范式转移 | `python/evolution/paradigm_shift.py` |
| 拓扑相变 | `python/evolution/topology.py` |
| 遗传沙盒 | `python/evolution/genetic_sandbox.py` |
