"""日志配置"""
import logging
import sys
from pathlib import Path

# Windows: 强制 StreamHandler 使用 UTF-8，避免 emoji/中文 触发 GBK 编码错误
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

stream_handler = logging.StreamHandler()
stream_handler.setStream(open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "tech_agent.log", encoding="utf-8"),
        stream_handler,
    ]
)
logger = logging.getLogger("tech_agent")
