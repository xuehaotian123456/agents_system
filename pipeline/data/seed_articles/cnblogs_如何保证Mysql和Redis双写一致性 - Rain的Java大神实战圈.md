# 如何保证Mysql和Redis双写一致性

> 来源: 博客园

如何保证Mysql和Redis双写一致性
这个问题本质上是
分布式系统中的数据一致性问题
。因为 MySQL 和 Redis 是两个独立的存储系统，无法做到原子性更新，所以我们只能通过合理的更新策略来
尽可能保证最终一致性
，同时兼顾性能和可用性。
先明确：哪些方案是绝对不能用的 ❌
很多人一开始会踩这些坑，我先排除掉：
错误方案
致命问题
先更新 Redis，再更新 MySQL
Redis 更新成功，MySQL 更新失败 → 数据永久不一致
先更新 MySQL，再更新 Redis
并发场景下会出现 "写覆盖" 问题，导致脏数据
双写都加分布式锁
性能极差，完全失去了 Redis 缓存的意义
业界主流的 4 种正确方案对比 📊
方案 1：先更新数据库，再删除缓存（最常用）
优点
：实现简单，性能好，出现不一致的概率极低
缺点
：极端情况下仍有不一致风险（数据库更新成功，删除缓存失败）
适用场景
：90% 以上的业务场景都可以用这个方案
方案 2：先删除缓存，再更新数据库
优点
：比 "先更库再删缓存" 更安全
缺点
：并发读场景下会出现 "缓存击穿" 问题
解决办法
：采用 "延迟双删" 策略
方案 3：更新数据库 + 消息队列异步删除缓存
优点
：解决了 "删除缓存失败" 的问题，有重试机制
缺点
：引入了 MQ 的复杂度，有一定的延迟
适用场景
：对一致性要求较高的业务
方案 4：基于 MySQL binlog 的最终一致性方案（终极方案）
优点
：完全解耦，业务代码无侵入，一致性最高
缺点
：架构最复杂，运维成本高
适用场景
：大型互联网公司，高并发高一致性要求的核心业务
面试必问：极端场景分析 🔍
场景 1：为什么是 "删除缓存" 而不是 "更新缓存"？
✅
答案
：
并发写场景下，更新缓存会出现 "写覆盖" 问题
很多缓存值不是简单的数据库字段映射，计算成本高
采用 "懒加载" 思想，只有当缓存被读取时才会重新计算，节省资源
场景 2："先更库再删缓存" 的极端不一致情况
发生条件
：
缓存刚好失效
线程 A 查询数据库，得到旧值
线程 B 更新数据库，然后删除缓存
线程 A 将旧值写入缓存
结果
：缓存中永远是旧数据，直到下一次更新或过期
解决办法
：
给缓存设置合理的过期时间（兜底方案）
采用 "延迟双删" 策略
使用 binlog 异步删除方案
我的生产环境最佳实践 ✨
基础方案
：先更新 MySQL，再删除 Redis 缓存
兜底方案
：所有缓存都设置过期时间（15 分钟 - 2 小时）
增强方案
：删除缓存失败时，通过 MQ 进行重试
终极方案
：核心业务使用 Canal 监听 binlog 异步更新缓存
生产级核心代码实现 🚀
基于 Spring Boot 3.x + Redis 7.x + RabbitMQ 3.x
1 基础方案：先更库再删缓存（带异常重试）
技术亮点
：
统一异常处理，删除失败立即重试 1 次
异步删除不阻塞主业务流程
日志埋点便于问题排查
@Service
@Slf4j
public class UserService {
    @Autowired
    private UserMapper userMapper;
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    // 自定义线程池，避免使用默认线程池导致OOM
    @Autowired
    private ThreadPoolTaskExecutor cacheExecutor;

    /**
     * 更新用户信息（基础双写方案）
     */
    @Transactional(rollbackFor = Exception.class)
    public void updateUser(User user) {
        // 1. 先更新数据库
        int rows = userMapper.updateById(user);
        if (rows == 0) {
            log.warn("更新用户信息失败，用户不存在: {}", user.getId());
            return;
        }

        // 2. 异步删除缓存，不阻塞主流程
        String cacheKey = "user:info:" + user.getId();
        cacheExecutor.execute(() -> {
            try {
                redisTemplate.delete(cacheKey);
                log.info("删除缓存成功: {}", cacheKey);
            } catch (Exception e) {
                // 立即重试1次，仍失败则记录告警，后续由定时任务兜底
                log.error("第一次删除缓存失败，重试中: {}", cacheKey, e);
                try {
                    redisTemplate.delete(cacheKey);
                    log.info("重试删除缓存成功: {}", cacheKey);
                } catch (Exception ex) {
                    log.error("重试删除缓存失败，需人工介入: {}", cacheKey, ex);
                    // 发送告警（邮件/短信/钉钉）
                    alertService.sendAlert("缓存删除失败", cacheKey);
                }
            }
        });
    }
}
2 增强方案：延迟双删（解决并发读写脏数据）
技术亮点
：
使用线程池实现延迟任务，不阻塞主线程
可配置延迟时间，适配不同数据库同步延迟
幂等性检查，避免重复删除
@Service
@Slf4j
public class UserService {
    // 省略其他注入...
    @Value("${cache.delay-delete-time:500}")
    private long delayDeleteTime;

    /**
     * 更新用户信息（延迟双删方案）
     */
    @Transactional(rollbackFor = Exception.class)
    public void updateUserWithDelayDelete(User user) {
        String cacheKey = "user:info:" + user.getId();
        
        // 1. 第一次删除缓存
        redisTemplate.delete(cacheKey);
        
        // 2. 更新数据库
        userMapper.updateById(user);
        
        // 3. 延迟删除缓存（核心：等待读线程完成旧值写入）
        cacheExecutor.schedule(() -> {
            // 幂等性检查：如果缓存不存在，无需删除
            if (Boolean.TRUE.equals(redisTemplate.hasKey(cacheKey))) {
                redisTemplate.delete(cacheKey);
                log.info("延迟删除缓存成功: {}", cacheKey);
            }
        }, delayDeleteTime, TimeUnit.MILLISECONDS);
    }
}
3 高可靠方案：MQ 异步删除（解决删除失败问题）
技术亮点
：
消息持久化 + 重试机制，保证最终一致性
幂等性设计，防止重复消费
死信队列处理失败消息，避免消息丢失
// 生产者
@Service
@Slf4j
public class CacheDeleteProducer {
    @Autowired
    private RabbitTemplate rabbitTemplate;

    public void sendDeleteMessage(String cacheKey) {
        try {
            // 消息体包含唯一ID，用于幂等性
            CacheDeleteMessage message = new CacheDeleteMessage(
                UUID.randomUUID().toString(),
                cacheKey
            );
            rabbitTemplate.convertAndSend("cache-exchange", "cache.delete", message);
            log.info("发送删除缓存消息成功: {}", message);
        } catch (Exception e) {
            log.error("发送删除缓存消息失败: {}", cacheKey, e);
            throw new RuntimeException("发送缓存删除消息失败", e);
        }
    }
}

// 消费者
@Component
@Slf4j
public class CacheDeleteConsumer {
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    @RabbitListener(queues = "cache-delete-queue")
    public void handleDeleteMessage(CacheDeleteMessage message) {
        String cacheKey = message.getCacheKey();
        String messageId = message.getMessageId();
        
        // 1. 幂等性检查：如果该消息已处理过，直接返回
        String idempotentKey = "cache:delete:idempotent:" + messageId;
        if (Boolean.TRUE.equals(redisTemplate.hasKey(idempotentKey))) {
            log.info("消息已处理，跳过: {}", messageId);
            return;
        }

        try {
            // 2. 删除缓存
            redisTemplate.delete(cacheKey);
            log.info("消费消息删