import asyncio

# 定义异步函数
async def work():
    print("任务开始，等待2秒")
    # await：等待异步IO，不阻塞全局
    await asyncio.sleep(2)
    print("任务结束")

# 程序入口，启动异步函数
async def main():
    # 在async函数内部，用await执行异步任务
    await work()

# 固定模板：脚本启动异步程序
if __name__ == "__main__":
    asyncio.run(main())