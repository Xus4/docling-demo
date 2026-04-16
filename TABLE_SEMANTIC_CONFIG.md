# 表格语义增强 - 大模型参数配置指南

## 概述

本项目支持通过环境变量精细控制大模型调用的各项参数，以优化表格语义增强的效果。

## 可用的配置参数

### 基础配置

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `TABLE_SEMANTIC_ENABLE` | bool | false | 是否启用表格语义增强 |
| `TABLE_SEMANTIC_BASE_URL` | string | - | 大模型API根地址 |
| `TABLE_SEMANTIC_API_KEY` | string | - | API密钥（可选） |
| `TABLE_SEMANTIC_MODEL` | string | - | 模型名称 |
| `TABLE_SEMANTIC_TIMEOUT_SEC` | float | 120.0 | 请求超时时间（秒） |
| `TABLE_SEMANTIC_MAX_CONCURRENCY` | int | 4 | 最大并发请求数 |

### 生成控制参数 ⭐ 新增

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `TABLE_SEMANTIC_MAX_TOKENS` | int | None | 最大生成token数。控制输出长度，建议值：2000-8000 |
| `TABLE_SEMANTIC_TEMPERATURE` | float | None | 温度参数(0-2)。控制随机性：0=最确定，2=最随机，建议值：0.3-0.8 |

### 上下文配置

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `TABLE_SEMANTIC_CONTEXT_BEFORE_CHARS` | int | None | 表格前上下文字符数（0表示不限制） |
| `TABLE_SEMANTIC_CONTEXT_AFTER_CHARS` | int | None | 表格后上下文字符数（0表示不限制） |

### 高级配置

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `TABLE_SEMANTIC_ON_ERROR` | string | skip | 错误处理策略：skip(跳过) 或 fail(失败) |
| `TABLE_SEMANTIC_THINKING_ENABLE` | bool | false | 是否启用思维链（Ollama兼容） |

## 使用示例

### 示例1：精确控制输出长度和质量

```bash
# .env 文件配置
TABLE_SEMANTIC_ENABLE=true
TABLE_SEMANTIC_BASE_URL=http://127.0.0.1:11434
TABLE_SEMANTIC_MODEL=qwen3.5:35b-a3b
TABLE_SEMANTIC_MAX_TOKENS=4000
TABLE_SEMANTIC_TEMPERATURE=0.7
```

### 示例2：低随机性，适合结构化数据

```bash
TABLE_SEMANTIC_ENABLE=true
TABLE_SEMANTIC_MODEL=qwen3.5:35b-a3b
TABLE_SEMANTIC_MAX_TOKENS=3000
TABLE_SEMANTIC_TEMPERATURE=0.3
```

### 示例3：高多样性，适合复杂表格

```bash
TABLE_SEMANTIC_ENABLE=true
TABLE_SEMANTIC_MODEL=qwen3.5:35b-a3b
TABLE_SEMANTIC_MAX_TOKENS=6000
TABLE_SEMANTIC_TEMPERATURE=0.8
TABLE_SEMANTIC_TOP_P=0.95
TABLE_SEMANTIC_FREQUENCY_PENALTY=0.5
```

## 参数调优建议

### Temperature（温度）
- **0.0-0.3**: 非常确定，适合数值型、事实性表格
- **0.4-0.7**: 平衡模式，适合大多数场景（推荐）
- **0.8-1.0**: 更有创意，适合需要推断的复杂表格
- **>1.0**: 高度随机，一般不推荐用于表格分析

### Max Tokens（最大token）
- **2000-3000**: 简单表格，少量数据
- **4000-5000**: 中等复杂度表格（推荐默认值）
- **6000-8000**: 复杂表格，包含大量数据和关系

### Top P（核采样）
- **0.7-0.85**: 更聚焦，减少不相关输出
- **0.85-0.95**: 平衡模式（推荐）
- **>0.95**: 更多样化，但可能引入不相关内容

### Presence Penalty（存在惩罚）
- **0.0**: 无惩罚（默认）
- **0.2-0.4**: 轻微惩罚，减少话题重复
- **0.5-0.8**: 中等惩罚，适合长表格避免重复描述

### Frequency Penalty（频率惩罚）
- **0.0**: 无惩罚（默认）
- **0.3-0.6**: 轻微惩罚，减少相同词汇重复
- **0.7-1.0**: 中等惩罚，提高表达多样性

## 注意事项

1. **参数留空**：如果不设置某个参数（留空或不配置），将使用模型API的默认值
2. **兼容性**：并非所有模型都支持所有参数，请参考您使用的大模型API文档
3. **Ollama特殊说明**：使用Ollama时，`thinking` 参数会被显式传入（true/false）
4. **性能影响**：`max_tokens` 越大，响应时间可能越长，token消耗也越大
5. **质量平衡**：建议先使用默认值测试，然后根据输出质量逐步调整参数

## 调试技巧

1. 启用详细日志查看实际发送的请求参数：
   ```bash
   DEBUG=true
   ```

2. 查看日志文件中的 `table_semantic.llm.request` 事件，确认参数是否正确传递

3. 使用较小的测试文件快速迭代参数配置

## 完整配置示例

```bash
# .env 文件
TABLE_SEMANTIC_ENABLE=true
TABLE_SEMANTIC_BASE_URL=http://127.0.0.1:11434
TABLE_SEMANTIC_API_KEY=your-api-key
TABLE_SEMANTIC_MODEL=qwen3.5:35b-a3b
TABLE_SEMANTIC_TIMEOUT_SEC=600
TABLE_SEMANTIC_MAX_CONCURRENCY=2
TABLE_SEMANTIC_MAX_TOKENS=4000
TABLE_SEMANTIC_TEMPERATURE=0.7
TABLE_SEMANTIC_TOP_P=0.9
TABLE_SEMANTIC_PRESENCE_PENALTY=0.0
TABLE_SEMANTIC_FREQUENCY_PENALTY=0.0
TABLE_SEMANTIC_CONTEXT_BEFORE_CHARS=0
TABLE_SEMANTIC_CONTEXT_AFTER_CHARS=0
TABLE_SEMANTIC_ON_ERROR=skip
TABLE_SEMANTIC_THINKING_ENABLE=false
```

## 技术支持

如遇到问题，请查看：
- 日志文件：`logs/webapp_*.log`
- 测试用例：`tests/test_llm_client.py` 和 `tests/test_table_augment.py`
- 源代码：`src/table_semantic/llm_client.py`
