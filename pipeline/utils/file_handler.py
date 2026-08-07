"""文件处理工具"""
import os
import hashlib
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

def get_file_md5(filepath: str) -> str | None:
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return None
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

def load_markdown(filepath: str) -> list[Document]:
    """加载 Markdown 文件，自动尝试多种编码"""
    for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
        try:
            return TextLoader(filepath, encoding=enc).load()
        except (UnicodeDecodeError, Exception):
            continue
    # 最后一次尝试
    return TextLoader(filepath, encoding="utf-8").load()

def list_files_by_type(directory: str, extensions: tuple[str]) -> list[str]:
    files = []
    if not os.path.isdir(directory):
        return files
    for f in os.listdir(directory):
        if f.endswith(extensions):
            files.append(os.path.join(directory, f))
    return files
