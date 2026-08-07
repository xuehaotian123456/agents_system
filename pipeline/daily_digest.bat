@echo off
REM DevPilot 每日技术摘要 — 用于 Windows 任务计划程序
REM 设置: 任务计划程序 → 创建任务 → 触发器: 每天 8:00 → 操作: 运行此 bat

cd /d E:\track_exp\tech_agent

REM 激活 conda 环境并运行
call D:\anaconda_1\Scripts\activate.bat rag_env
python services/digest_mail.py --crawl --send
