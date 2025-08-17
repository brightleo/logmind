# LogMind - 本地日志分析助手

LogMind 是一个基于 PyQt5 和大语言模型 (LLM) 的本地日志分析工具。它可以帮助开发者快速分析错误日志，定位问题根因，并提供修复建议。

## 功能特点

- **多语言支持**：支持分析多种编程语言的代码，包括 Java、Python、JavaScript、TypeScript、C++、C、C#、Go、PHP、Ruby、Swift、Kotlin、Rust 等。
- **AI 驱动分析**：集成大语言模型（兼容 Ollama 和 OpenAI API），提供智能日志分析和代码审查。
- **本地化处理**：所有分析过程在本地进行，保护您的代码隐私。
- **直观的图形界面**：基于 PyQt5 构建的用户友好的图形界面。
- **灵活的配置**：支持本地模型（如 Ollama）和远程模型（如 OpenAI、DeepSeek）。
- **代码上下文关联**：可以关联相关代码文件，提供更精准的分析。
- **代理支持**：支持通过代理连接 AI 模型服务。

## 界面预览

![LogMind 界面](screenshots/logmind_ui.png) *(运行 logmind.py 后的界面截图)*

## 安装说明

### 环境要求

- Python 3.8 或更高版本
- PyQt5
- openai Python 库
- httpx

### 安装步骤

1. 克隆项目仓库：
   ```bash
   git clone https://github.com/brightleo/logmind.git
   cd logmind
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   
   或者手动安装依赖：
   ```bash
   pip install PyQt5 openai httpx
   ```

3. 安装并配置本地 AI 模型（可选，推荐 Ollama）：
   - 安装 [Ollama](https://ollama.com/)
   - 拉取模型，例如 Qwen2 或 Llama3：
     ```bash
     ollama pull qwen2
     # 或
     ollama pull llama3
     ```

## 使用方法

1. 运行 LogMind：
   ```bash
   python logmind.py
   ```

2. 在图形界面中：
   - 在“问题描述”框中描述您遇到的问题
   - 在“错误日志”框中粘贴错误日志
   - （可选）点击 "+" 按钮添加包含相关代码的文件夹
   - 点击“分析日志”按钮开始分析
   - 查看右侧“分析结果”区域的分析报告

## 配置说明

LogMind 会自动生成配置文件 `logmind_config.json`，您可以通过界面中的“设置”按钮或直接编辑该文件来配置：

- **AI 模型**：选择本地模型（Ollama）或远程模型（OpenAI 兼容 API）
- **模型参数**：设置模型名称、API 地址、API 密钥等
- **分析参数**：设置温度 (temperature) 和最大令牌数 (max_tokens)
- **代理设置**：如果需要通过代理连接 AI 服务，可配置代理主机、端口、用户名和密码

### 示例配置 (logmind_config.json)

```json
{
  "ai_config": {
    "model_type": "local",
    "local": {
      "base_url": "http://localhost:11434/v1",
      "api_key": "sk-no-key-required",
      "model_name": "qwen2"
    },
    "remote": {
      "base_url": "https://api.openai.com/v1",
      "api_key": "your-openai-api-key",
      "model_name": "gpt-4-turbo"
    },
    "analysis_params": {
      "temperature": 0.1,
      "max_tokens": 2000
    }
  },
  "proxy_config": {
    "enabled": false,
    "host": "",
    "port": "",
    "username": "",
    "password": ""
  }
}
```

## 技术栈

- [Python](https://www.python.org/)
- [PyQt5](https://pypi.org/project/PyQt5/)
- [OpenAI Python API library](https://github.com/openai/openai-python)
- [httpx](https://www.python-httpx.org/)

## 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。
