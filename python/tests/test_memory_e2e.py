"""FourDMem 记忆系统端到端测试脚本

测试完整链路: 写入 → 检索 → 反馈 → 持久化
运行方式: python test_memory_e2e.py

"""
import json
import sqlite3
import sys
import os
import tempfile

# 确保能找到 fourdmem 模块
sys.path.insert(0, os.path.dirname(__file__))

import fourdmem

# Use temp directory to avoid Tantivy lock conflicts with other tests
DB_PATH = os.path.join(tempfile.mkdtemp(prefix="fourdmem_e2e_"), "evidence.db")

def test_memory_chain():
    """完整测试: 写入 → 检索 → 反馈 → 持久化"""
    
    print("=" * 50)
    print("FourDMem 记忆系统端到端测试")
    print("=" * 50)
    
    # 1. 初始化引擎
    print("\n[1] 初始化引擎...")
    engine = fourdmem.FourDMemEngine(DB_PATH)
    wake_result = json.loads(engine.wake_up())
    print(f"    状态: {wake_result['status']}")
    print(f"    当前 L0 证据数: {wake_result['memory_stats']['l0_evidence']}")
    
    # 2. 写入记忆
    print("\n[2] 写入测试记忆...")
    test_memories = [
        ("session-test", "user", "Rust 负责图引擎和全文检索，Python 负责认知逻辑"),
        ("session-test", "assistant", "明白了，这是混合架构项目"),
        ("session-test", "user", "数据库路径是 G:\\DeepSeek\\FourDMem\\data\\vault\\evidence.db"),
        ("session-test", "tool", "编译成功，所有测试通过"),
    ]
    
    for session_id, role, content in test_memories:
        result = json.loads(engine.save(session_id, role, content, "{}"))
        print(f"    已保存: [{role}] {content[:40]}...")
    
    # 3. 检索记忆
    print("\n[3] 检索记忆...")
    queries = ["Rust 架构", "数据库路径", "编译测试"]
    
    for query in queries:
        result = json.loads(engine.query(query, 5))
        print(f"\n    查询: '{query}'")
        print(f"    结果数: {len(result['results'])}")
        if result['results']:
            for r in result['results'][:2]:
                print(f"      - {r['content'][:60]}...")
        else:
            print("      (无结果)")
    
    # 4. 反馈打分
    print("\n[4] 反馈打分...")
    engine.feedback("Rust", 0.8)
    print("    已对 'Rust' 相关记忆打分 +0.8")
    
    # 5. 验证持久化
    print("\n[5] 验证持久化 (直接查 SQLite)...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM evidence")
    count = cursor.fetchone()[0]
    print(f"    L0 证据总行数: {count}")
    
    cursor = conn.execute(
        "SELECT id, role, substr(content, 1, 60) FROM evidence ORDER BY id DESC LIMIT 5"
    )
    for row in cursor.fetchall():
        print(f"      [{row[0]}] {row[1]}: {row[2]}...")
    conn.close()
    
    # 6. 推进时间
    print("\n[6] 推进主观时间...")
    tick = engine.advance_tick()
    print(f"    当前 tick: {tick}")
    
    # 7. 元认知反思
    print("\n[7] 元认知反思...")
    reflect_result = json.loads(engine.reflect("Rust 架构", 3, 0.7))
    print(f"    置信度: {reflect_result['confidence']:.2f}")
    print(f"    建议深挖: {reflect_result['should_drill_down']}")
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)

if __name__ == "__main__":
    test_memory_chain()
