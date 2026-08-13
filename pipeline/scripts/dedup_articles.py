"""
文章去重 — 内容 hash 级去重 (数据质量治理)
=========================================
问题: 同一篇文章被不同爬取轮次以不同前缀保存多次
  (juejin_ / sched_juejin_ / digest_juejin_ / batch_juejin_ / auto_juejin_),
  语料库中 4,331 chunks 里有 433 个内容重复 (10%)。
  重复内容浪费检索 top-k 槽位, 干扰评测。

策略:
  1. 规范化: 去掉 "> 爬取时间" 等易变 header 后计算内容 hash
  2. 按来源优先级保留: gitee(权威) > juejin/cnblogs(原创) > 其他(定时爬取副本)
  3. 删除重复文件 (移入 data/duplicates/ 备份)

运行: python scripts/dedup_articles.py
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ARTICLES_DIR = Path(__file__).parent.parent / "data" / "articles"
BACKUP_DIR = Path(__file__).parent.parent / "data" / "duplicates"

# 保留优先级 (越小越优先保留)
SOURCE_PRIORITY = {
    "gitee": 0,
    "juejin": 1,
    "cnblogs": 1,
    "ai_ref": 0,
    "auto": 3,
    "batch": 3,
    "digest": 3,
    "sched": 2,
}


def _normalize_content(content: str) -> str:
    """去掉易变行 (爬取时间/URL) 后规范化, 用于内容 hash"""
    lines = []
    for line in content.split("\n"):
        if line.startswith("> 爬取时间") or line.startswith("> URL"):
            continue
        lines.append(line.strip())
    return "\n".join(lines).strip()


def _content_hash(content: str) -> str:
    return hashlib.md5(_normalize_content(content).encode("utf-8")).hexdigest()


def main():
    if not ARTICLES_DIR.exists():
        print("articles 目录不存在")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(ARTICLES_DIR.glob("*.md"))
    seen: dict[str, Path] = {}   # hash -> 保留的文件
    duplicates: list[Path] = []

    for f in files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        h = _content_hash(content)
        existing = seen.get(h)
        if existing is None:
            seen[h] = f
            continue
        # 比较来源优先级: 当前文件优先级更高则替换保留者
        cur_prio = SOURCE_PRIORITY.get(f.name.split("_")[0], 9)
        ex_prio = SOURCE_PRIORITY.get(existing.name.split("_")[0], 9)
        if cur_prio < ex_prio:
            duplicates.append(existing)
            seen[h] = f
        else:
            duplicates.append(f)

    print(f"总文件: {len(files)}, 保留: {len(seen)}, 重复: {len(duplicates)}")

    for dup in duplicates:
        target = BACKUP_DIR / dup.name
        shutil.move(str(dup), str(target))
        print(f"  [去重] {dup.name[:60]}")

    print(f"\n完成: {len(seen)} 篇唯一文章, {len(duplicates)} 篇移入 data/duplicates/")
    print("注意: 去重后需重建向量库和 KG:")
    print("  python -c \"from rag.vector_store import VectorStore; vs=VectorStore(); vs.load_articles(force_rebuild=True)\"")


if __name__ == "__main__":
    main()
