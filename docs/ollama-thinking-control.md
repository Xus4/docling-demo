# Ollama 思考模式开关控制说明

本文档记录当前项目在调用 Ollama OpenAI 兼容接口时，如何控制模型的思考模式。可直接复用到 Java 项目中。

## 适用范围

适用于通过 Ollama 的 OpenAI 兼容接口调用模型：

```text
http://127.0.0.1:11434/v1/chat/completions
```

也就是基础地址通常配置为：

```text
http://127.0.0.1:11434/v1
```

## 核心结论

Ollama 不识别百炼/Qwen 风格的 `enable_thinking` 和 `max_reasoning_tokens` 作为思考模式控制字段。

针对 Ollama，关闭思考模式应在 OpenAI compatible 请求体中加入：

```json
{
  "reasoning_effort": "none"
}
```

当前项目为了兼容百炼和 Ollama，会在关闭 thinking 时同时发送：

```json
{
  "enable_thinking": false,
  "max_reasoning_tokens": 0,
  "reasoning_effort": "none"
}
```

其中真正对 Ollama 起作用的是 `reasoning_effort: "none"`。

## 配置建议

建议 Java 项目保留两个配置项：

```yaml
llm:
  base-url: http://127.0.0.1:11434/v1
  enable-thinking: false
  max-reasoning-tokens: 256
```

含义：

| 配置 | 含义 |
|---|---|
| `llm.base-url` | LLM 服务地址。Ollama OpenAI 兼容接口一般以 `/v1` 结尾 |
| `llm.enable-thinking` | 是否允许模型思考模式 |
| `llm.max-reasoning-tokens` | 思考 token 上限。对 Ollama 不是主要控制字段，但可保留给 Qwen/百炼兼容 |

## 判断是否是 Ollama

当前项目采用保守判断：

1. `baseUrl` 以 `/v1` 结尾。
2. URL 中包含 `11434`。
3. URL 不是 DashScope/百炼地址。

Java 可参考：

```java
static boolean isLikelyOllamaOpenAiBaseUrl(String baseUrl) {
    if (baseUrl == null || baseUrl.isBlank()) {
        return false;
    }
    String u = baseUrl.trim().toLowerCase();
    while (u.endsWith("/")) {
        u = u.substring(0, u.length() - 1);
    }
    if (u.contains("dashscope") || u.contains("compatible-mode")) {
        return false;
    }
    return u.endsWith("/v1") && u.contains("11434");
}
```

## 请求体构造规则

### 开启 thinking

如果 `enableThinking=true`，Ollama 请求体不需要加 `reasoning_effort: "none"`：

```json
{
  "model": "qwen3.5:9b",
  "messages": [
    {
      "role": "user",
      "content": "请把这段文字整理为 Markdown"
    }
  ],
  "temperature": 0.0,
  "max_tokens": 4096
}
```

### 关闭 thinking

如果 `enableThinking=false`，且判断为 Ollama，则加：

```json
{
  "model": "qwen3.5:9b",
  "messages": [
    {
      "role": "user",
      "content": "请把这段文字整理为 Markdown"
    }
  ],
  "temperature": 0.0,
  "max_tokens": 4096,
  "enable_thinking": false,
  "max_reasoning_tokens": 0,
  "reasoning_effort": "none"
}
```

## Java Map 示例

```java
Map<String, Object> payload = new LinkedHashMap<>();
payload.put("model", model);
payload.put("messages", messages);

if (temperature != null) {
    payload.put("temperature", temperature);
}
if (maxTokens != null) {
    payload.put("max_tokens", maxTokens);
}

boolean thinkingOn = enableThinking && !forceDisableThinking;

if (forceDisableThinking) {
    payload.put("enable_thinking", false);
    payload.put("max_reasoning_tokens", 0);
} else {
    payload.put("enable_thinking", enableThinking);
    if (enableThinking && maxReasoningTokens != null) {
        payload.put("max_reasoning_tokens", maxReasoningTokens);
    } else if (!enableThinking) {
        payload.put("max_reasoning_tokens", 0);
    }
}

if (isLikelyOllamaOpenAiBaseUrl(baseUrl) && !thinkingOn) {
    payload.put("reasoning_effort", "none");
}
```

## 响应处理规则

Ollama/Qwen thinking 模式下，有时会出现：

```json
{
  "choices": [
    {
      "message": {
        "content": "",
        "thinking": "..."
      }
    }
  ]
}
```

或：

```json
{
  "choices": [
    {
      "message": {
        "content": "",
        "reasoning_content": "..."
      }
    }
  ]
}
```

当前项目的处理原则：

1. `message.content` 才是可交付正文。
2. `message.thinking` 和 `message.reasoning_content` 不回填为正文。
3. 如果 `content` 为空但 thinking/reasoning 非空，判定为 reasoning-only。
4. 下一轮重试强制关闭 thinking，并带上 `reasoning_effort: "none"`。

Java 可参考：

```java
static boolean isReasoningOnlyResponse(Map<String, Object> response) {
    List<?> choices = (List<?>) response.get("choices");
    if (choices == null || choices.isEmpty()) {
        return false;
    }

    Object first = choices.get(0);
    if (!(first instanceof Map<?, ?> choice)) {
        return false;
    }

    Object messageObj = choice.get("message");
    if (!(messageObj instanceof Map<?, ?> message)) {
        return false;
    }

    Object content = message.get("content");
    boolean hasContent = content instanceof String s && !s.isBlank();

    Object reasoning = message.get("reasoning_content");
    boolean hasReasoning = reasoning instanceof String s && !s.isBlank();

    Object thinking = message.get("thinking");
    boolean hasThinking = thinking instanceof String s && !s.isBlank();

    return !hasContent && (hasReasoning || hasThinking);
}
```

## 重试策略建议

建议最多重试 2 到 3 次：

1. 第一次按配置发送。
2. 如果返回 `content` 非空，直接使用。
3. 如果返回 reasoning-only，下一次请求设置 `forceDisableThinking=true`。
4. 如果多次仍然 `content` 为空，抛错，不要把 thinking/reasoning 当正文。

伪代码：

```java
boolean lastReasoningOnly = false;

for (int attempt = 1; attempt <= maxAttempts; attempt++) {
    boolean forceDisableThinking = attempt > 1 && lastReasoningOnly;

    Map<String, Object> payload = buildPayload(
        model,
        messages,
        enableThinking,
        forceDisableThinking,
        maxReasoningTokens,
        baseUrl
    );

    Map<String, Object> response = postChatCompletions(payload);
    String content = extractAssistantContent(response);
    if (content != null && !content.isBlank()) {
        return content;
    }

    lastReasoningOnly = isReasoningOnlyResponse(response);
}

throw new IllegalStateException("LLM assistant.content is empty after retries");
```

## 测试点

建议 Java 项目至少覆盖这些单元测试：

1. Ollama + `enableThinking=false` 时，请求体包含 `reasoning_effort=none`。
2. Ollama + `enableThinking=true` 时，请求体不包含 `reasoning_effort=none`。
3. DashScope/百炼地址关闭 thinking 时，不添加 `reasoning_effort=none`。
4. `content=""` 且 `thinking` 非空时，判定为 reasoning-only。
5. reasoning-only 后下一轮请求强制关闭 thinking。

## 注意事项

- `enable_thinking` 和 `max_reasoning_tokens` 是为了兼容 Qwen/百炼风格，不应依赖它们控制 Ollama。
- Ollama 的核心关闭字段是 `reasoning_effort: "none"`。
- 不要把 `thinking` 或 `reasoning_content` 当作最终正文，否则会把模型内部思考链写进业务结果。
- 如果使用非 `11434` 端口部署 Ollama，当前项目的启发式判断需要调整，建议增加显式配置，例如 `llm.provider=ollama`。
