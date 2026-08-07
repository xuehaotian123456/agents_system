"""知识图谱可视化 — 使用 pyvis 生成交互式 HTML 图"""
from pathlib import Path
from pyvis.network import Network
from utils.logger_handler import logger

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "graphs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _ensure_utf8(fpath: str):
    """确保 HTML 文件是 UTF-8 编码（pyvis 在 Windows 上可能写 GBK）"""
    for enc in ["utf-8", "gbk", "gb2312"]:
        try:
            with open(fpath, "r", encoding=enc) as f:
                content = f.read()
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            return
        except (UnicodeDecodeError, UnicodeError):
            continue

COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
    "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#8e44ad",
    "#27ae60", "#d35400", "#16a085", "#2c3e50", "#f1c40f",
]

def build_entity_graph(entity_name: str, depth: int = 1) -> str | None:
    """为中心实体构建交互式关系图"""
    try:
        from rag.knowledge_graph import get_kg
        kg = get_kg()
        if not kg.is_built:
            return None

        info = kg.get_entity(entity_name)
        if not info:
            return None

        net = Network(height="400px", width="100%", bgcolor="#ffffff",
                      font_color="#333333", directed=False)
        net.heading = f"知识图谱: {entity_name}"

        # 中心节点
        center_freq = info.get("freq", 1)
        center_size = max(25, min(55, 20 + center_freq * 3))
        net.add_node(entity_name, label=entity_name,
                     title=f"频次: {center_freq} 次",
                     size=center_size, color="#e74c3c", shape="dot")

        # 直接邻居
        color_idx = 0
        related = info.get("related", [])
        for rel in related[:8]:
            neighbor = rel["entity"]
            co_occur = rel["co_occur"]
            nsize = max(12, min(35, 12 + co_occur * 2))
            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1
            net.add_node(neighbor, label=neighbor,
                         title=f"与 {entity_name} 共现: {co_occur} 次",
                         size=nsize, color=color)
            net.add_edge(entity_name, neighbor, value=co_occur,
                         title=f"共现 {co_occur} 次")

        # 保存
        safe_name = "".join(c for c in entity_name if c.isalnum() or c in (' ', '-', '_')).strip()[:30]
        fname = f"kg_{safe_name}.html"
        fpath = str(OUTPUT_DIR / fname)
        net.save_graph(fpath)
        # pyvis 在 Windows 上可能用 GBK 编码，强制转 UTF-8
        _ensure_utf8(fpath)
        logger.info(f"[GraphViz] 实体图: {fname} ({len(net.nodes)} nodes, {len(net.edges)} edges)")
        return fpath
    except Exception as e:
        logger.error(f"[GraphViz] 失败: {e}")
        return None

def build_global_graph(top_n: int = 30) -> str | None:
    """全局知识图谱（Top N 实体）"""
    try:
        from rag.knowledge_graph import get_kg
        kg = get_kg()
        if not kg.is_built:
            return None

        net = Network(height="520px", width="100%", bgcolor="#ffffff",
                      font_color="#333333", directed=False)
        net.heading = "全局知识图谱"

        # Top N 实体
        sorted_entities = sorted(kg.entity_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        entity_set = {e for e, _ in sorted_entities}

        color_idx = 0
        for entity, freq in sorted_entities:
            nsize = max(12, min(45, 12 + freq * 2))
            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1
            net.add_node(entity, label=entity, title=f"频次: {freq}", size=nsize, color=color)

        # 关系边
        for e1 in entity_set:
            for e2 in entity_set:
                if e1 >= e2:
                    continue
                co = kg.co_occurrence.get(e1, {}).get(e2, 0)
                if co > 1:
                    net.add_edge(e1, e2, value=co, title=f"共现 {co} 次")

        fpath = str(OUTPUT_DIR / "kg_global.html")
        net.save_graph(fpath)
        _ensure_utf8(fpath)
        logger.info(f"[GraphViz] 全局图: {len(net.nodes)} nodes, {len(net.edges)} edges")
        return fpath
    except Exception as e:
        logger.error(f"[GraphViz] 全局图失败: {e}")
        return None
