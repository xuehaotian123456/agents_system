from agent.tools.rag_tools import rag_search, kg_lookup, save_article
from agent.tools.crawler_tools import fetch_article, trending_list
from agent.tools.utility_tools import search_web, trend_report, user_profile
from agent.tools.advanced_tools import daily_digest, push_daily_digest, compare_tech, code_example, create_tool, custom_search

ALL_TOOLS = [
    rag_search, kg_lookup, save_article,              # 知识工具
    fetch_article, trending_list,                      # 爬虫工具
    search_web, trend_report, user_profile,            # 信息+画像工具
    daily_digest, push_daily_digest,                   # 摘要工具
    compare_tech, code_example,                        # 高级工具
    create_tool, custom_search,                        # 动态工具
]
