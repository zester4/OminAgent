# 🤖 OmniAgent - Your Versatile AI Assistant

![Version](https://img.shields.io/badge/version-9.8-blue)
![Python](https://img.shields.io/badge/python-3.12-green)
![License](https://img.shields.io/badge/license-MIT-orange)

OmniAgent is a powerful, multi-functional AI assistant built with Google's Gemini model, designed to handle a wide range of tasks from code generation to image analysis.

## ✨ Key Features

- 🧠 **Advanced Language Model**: Powered by Gemini 2.0 Flash for fast, accurate responses
- 🔄 **Streaming Responses**: Real-time response generation with progress updates
- 💾 **Long-term Memory**: Vector database for semantic memory and context retention
- 🛠️ **Rich Tool Integration**: Multiple specialized tools for various tasks
- 📊 **Data Visualization**: Create various types of plots and charts
- 🖼️ **Image Generation**: Create images using Stability AI's powerful API
- 👁️ **Image Analysis**: AWS Rekognition integration for facial recognition and image analysis
- 📝 **File Handling**: Process various file types with MIME detection
- 🌐 **Web Integration**: Web scraping and search capabilities
- ⚡ **High Performance**: 1M token context window for handling large conversations

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.12+
Git
```

### 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/zester4/OmniAgent.git
cd OmniAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key
STABILITY_API_KEY=your_stability_api_key
OPENWEATHERMAP_API_KEY=your_weather_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
GITHUB_API_KEY=your_github_api_key
```

### 🎮 Usage

Run the agent:
```bash
python main.py
```

Run the web interface:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🛠️ Available Tools

### Core Tools
- 🔍 **Search Tool**: Web search for current information
- 🌐 **Web Scraper**: Extract content from websites
- 💻 **Code Execution**: Execute and manage Python code
- 🕒 **DateTime Tool**: Current date and time operations
- 📊 **Data Visualization**: Create various types of plots

### Integration Tools
- 🖼️ **Image Generation**: Create images using Stability AI
- 👁️ **AWS Rekognition**: Image analysis and facial recognition
- 🌡️ **Weather Tool**: Real-time weather information
- 📂 **GitHub Tool**: Manage repositories and files

### Memory & File Tools
- 🧠 **Vector Search**: Semantic memory search
- 📁 **File Processing**: Handle various file types
- 💾 **Memory Management**: Conversation history management

## 💬 Example Interactions

```
You: What's the weather in Tokyo?
OmniBot: Let me check that for you...
Weather in Tokyo: clear sky, Temp: 18°C (feels like 17°C), Humidity: 65%

You: Generate an image of a psychedelic tree
OmniBot: I'll create that image for you...
Image generated and saved to generated_images/psychedelic_tree.jpeg

You: Analyze this image for me
OmniBot: Let me look at that image...
[Provides detailed analysis of colors, objects, and composition]
```

## 🔐 Security

- All API keys are managed securely through environment variables
- File operations include safety checks and validation
- Secure error handling and logging mechanisms

## 📊 Performance

- 1M token context window
- Streaming responses for real-time interaction
- Efficient memory management with automatic pruning
- Vector database for fast semantic search

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for the language model
- Stability AI for image generation
- AWS Rekognition for image analysis
- All other open-source contributors

---

Built with ❤️ by Raiden Agents - Automating workflows through AI