# [Bug]: TypeError: __init__() got an unexpected keyword argument 'lm_shift_labels'

> 来源: gitee_issue/PaddlePaddle/PaddleNLP | 可信度: gitee_issue_labeled (0.8) | 标签: bug
> URL: https://gitee.com/paddlepaddle/PaddleNLP/issues/I8AN8R
> 爬取时间: 2026-08-12T12:18:21.030235

### 软件环境

```Markdown
- paddlepaddle:
- paddlepaddle-gpu: paddlepaddle-gpu 2.5.1
- paddlenlp: paddlenlp 2.6.1.post0

```

### 重复问题

- [x] I have searched the existing issues


### 错误描述

```Markdown
Traceback (most recent call last):
  File "/home/ubuntu/smyin/llama/finetune_generation.py", line 338, in <module>
    main()
  File "/home/ubuntu/smyin/llama/finetune_generation.py", line 141, in main
    model = model_class.from_pretrained(
  File "/home/ubuntu/.conda/envs/paddle_env/lib/python3.9/site-packages/paddlenlp/transformers/auto/modeling.py", line 837, in from_pretrained
    return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
  File "/home/ubuntu/.conda/envs/paddle_env/lib/python3.9/site-packages/paddlenlp/transformers/auto/modeling.py", line 391, in _from_pretrained
    return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
  File "/home/ubuntu/.conda/envs/paddle_env/lib/python3.9/site-packages/paddlenlp/transformers/model_utils.py", line 2073, in from_pretrained
    model = cls(config, *init_args, **model_kwargs)
  File "/home/ubuntu/.conda/envs/paddle_env/lib/python3.9/site-packages/paddlenlp/transformers/utils.py", line 255, in __impl__
    init_func(self, *args, **kwargs)
TypeError: __init__() got an unexpected keyword argument 'lm_shift_labels'
```

### 稳定复现步骤 & 代码

未对源码进行改动，直接运行的lora微调脚本

