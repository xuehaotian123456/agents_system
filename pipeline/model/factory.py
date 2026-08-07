"""LLM 工厂 — 主模型 + 备份降级"""
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from contextvars import ContextVar
from utils.config_handler import get_llm_config
from utils.logger_handler import logger
import time

llm_config = get_llm_config()
chat_model = ChatTongyi(model=llm_config.get("model_name", "deepseek-v3.2"))
embed_model = DashScopeEmbeddings(model="text-embedding-v4")

backup_model = None
try:
    backup_name = llm_config.get("backup_model_name")
    if backup_name:
        backup_model = ChatTongyi(model=backup_name)
        logger.info(f"备份模型已初始化: {backup_name}")
except Exception as e:
    logger.warning(f"备份模型初始化失败: {e}")

retry_cfg = llm_config.get("retry", {})
_last_model_used: ContextVar[str] = ContextVar("last_model", default="unknown")

@retry(
    stop=stop_after_attempt(retry_cfg.get("max_attempts", 3)),
    wait=wait_exponential(min=retry_cfg.get("min_wait", 2), max=retry_cfg.get("max_wait", 10)),
    retry=retry_if_exception_type(Exception),
    reraise=False
)
def _call_primary(prompt):
    start = time.time()
    response = chat_model.invoke(prompt)
    logger.debug(f"主模型调用成功: {time.time() - start:.2f}s")
    return response

def robust_llm_call(prompt):
    """带降级的 LLM 调用"""
    try:
        response = _call_primary(prompt)
        _last_model_used.set("primary")
        return response
    except Exception as e:
        logger.warning(f"主模型失败: {e}")
        if backup_model:
            try:
                response = backup_model.invoke(prompt)
                _last_model_used.set("backup")
                logger.info("降级到备份模型成功")
                return response
            except Exception as be:
                raise RuntimeError(f"主模型和备份模型均失败: 主={e}, 备份={be}")
        raise

def get_last_model() -> str:
    return _last_model_used.get()
