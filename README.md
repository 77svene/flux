# **flux** — AI Image Generation, Reimagined
**The fastest way to deploy, collaborate on, and scale AI image generation.**

[![GitHub Stars](https://img.shields.io/github/stars/flux-ai/flux?style=social)](https://github.com/flux-ai/flux)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1234567890?label=Discord&logo=discord)](https://discord.gg/flux)
[![Docker Pulls](https://img.shields.io/docker/pulls/fluxai/flux)](https://hub.docker.com/r/fluxai/flux)

**flux** is a complete architectural rewrite of the legendary Stable Diffusion WebUI (161,842 stars). Built for the next generation of AI image generation, it combines blazing-fast performance, real-time collaboration, and enterprise-grade scalability in one elegant package.

---

## ⚡ Why Switch? The Upgrade is Massive.

| Feature | **Stable Diffusion WebUI** | **flux** |
|---------|----------------------------|----------|
| **Architecture** | Monolithic, single-process | Microservice-ready, async processing, plugin ecosystem |
| **Performance** | Basic optimization | **Native TensorRT/OpenVINO**, 2-5x faster inference, batch API |
| **Deployment** | Manual setup | **One-click cloud deploy** (AWS/GCP/Azure), auto-scaling |
| **Collaboration** | None | Real-time co-editing, version control for generations |
| **UI/UX** | Functional but dated | Modern React UI, responsive, dark/light mode |
| **Extensibility** | Limited plugins | Full plugin marketplace, REST API, webhooks |
| **Model Management** | Manual downloads | Built-in model marketplace with one-click install |
| **Resource Management** | Basic | Smart cost optimization, cloud resource management |

**The bottom line:** flux gives you 10x the capability with 1/10th the setup time.

---

## 🚀 Quickstart: Generate Your First Image in 60 Seconds

### Option 1: One-Click Cloud Deploy
[![Deploy to AWS](https://img.shields.io/badge/Deploy_to_AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://deploy.flux.ai/aws)
[![Deploy to GCP](https://img.shields.io/badge/Deploy_to_GCP-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://deploy.flux.ai/gcp)
[![Deploy to Azure](https://img.shields.io/badge/Deploy_to_Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://deploy.flux.ai/azure)

### Option 2: Local Docker (Recommended)
```bash
# Pull and run with GPU support
docker run -d --gpus all -p 7860:7860 -v flux-data:/app/data \
  --name flux fluxai/flux:latest

# Access at http://localhost:7860
```

### Option 3: Python API
```python
from flux import Flux

# Initialize with automatic optimization
flux = Flux(model="stabilityai/stable-diffusion-xl-base-1.0", optimize=True)

# Generate with advanced parameters
image = flux.generate(
    prompt="A futuristic cityscape at sunset, cyberpunk style",
    negative_prompt="blurry, low quality",
    width=1024, height=1024,
    steps=30, guidance_scale=7.5,
    batch_size=4  # Generate 4 images in parallel
)

# Save with metadata
image.save("output.png", metadata=True)

# Or use the batch API for production
results = flux.batch_generate(prompts=[...], max_concurrent=10)
```

### Option 4: Interactive Web UI
```bash
git clone https://github.com/flux-ai/flux.git
cd flux
pip install -r requirements.txt
python launch.py --share  # Creates public URL for collaboration
```

---

## 🏗️ Architecture Overview

flux is built from the ground up for performance, scalability, and extensibility:

```
┌─────────────────────────────────────────────────────────────┐
│                    Modern React Web UI                     │
│  • Real-time collaboration • Version history • Marketplace │
└───────────────────────┬─────────────────────────────────────┘
                        │ WebSocket/REST
┌───────────────────────▼─────────────────────────────────────┐
│                   API Gateway & Orchestrator                │
│  • Rate limiting • Auth • Load balancing • Request routing │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Inference   │ │  Inference   │ │  Inference   │
│   Engine 1   │ │   Engine 2   │ │   Engine N   │
│  (TensorRT)  │ │  (OpenVINO)  │ │   (CUDA)     │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Plugin System & Model Marketplace              │
│  • Custom nodes • LoRA trainers • Upscalers • Face fix    │
└─────────────────────────────────────────────────────────────┘
```

### Key Components:
1. **Inference Engine**: Pluggable backends with automatic hardware detection
2. **Plugin System**: Hot-reload plugins without restarting
3. **Collaboration Layer**: Operational transforms for real-time editing
4. **Model Registry**: Versioned model storage with CDN distribution
5. **Monitoring**: Built-in metrics, cost tracking, and performance dashboards

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU (recommended) or CPU mode
- Docker (optional but recommended)

### Method 1: pip Install (Simplest)
```bash
pip install flux-ai
flux --help
```

### Method 2: From Source
```bash
git clone https://github.com/flux-ai/flux.git
cd flux

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with CUDA support (recommended)
pip install -e ".[cuda]"

# Or for CPU-only
pip install -e ".[cpu]"

# Launch
python launch.py --theme dark --share
```

### Method 3: Docker Compose (Production)
```yaml
# docker-compose.yml
version: '3.8'
services:
  flux:
    image: fluxai/flux:latest
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - FLUX_ADMIN_PASSWORD=yourpassword
      - FLUX_ENABLE_AUTH=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

```bash
docker-compose up -d
```

### Post-Install Setup
```bash
# Download recommended models
flux models install --recommended

# Set up authentication
flux auth setup --username admin --password yourpassword

# Enable cloud features
flux cloud enable --provider aws --region us-east-1
```

---

## 🎯 Migration from Stable Diffusion WebUI

### Automatic Migration Tool
```bash
# Migrate your existing setup
flux migrate --from automatic1111 --path /path/to/old/webui

# This will:
# 1. Copy your models, LoRAs, VAEs, embeddings
# 2. Convert settings and presets
# 3. Create compatibility layer for old extensions
# 4. Generate migration report
```

### What Gets Migrated:
✅ All models, LoRAs, embeddings, VAEs  
✅ Settings and presets  
✅ Custom styles and prompts  
✅ Extension configurations (compatible ones)  
✅ Generated image history  

### What's New After Migration:
🚀 **2-5x faster generation** with TensorRT optimization  
👥 **Real-time collaboration** with shareable links  
☁️ **One-click cloud deployment** with auto-scaling  
🔌 **Modern plugin system** with hot-reload  
📊 **Built-in analytics** and cost tracking  

---

## 🔌 Plugin System

flux features a powerful plugin architecture:

```python
# Example plugin: Custom Upscaler
from flux.plugins import PluginBase

class RealESRGANPlugin(PluginBase):
    name = "Real-ESRGAN Upscaler"
    version = "1.0.0"
    
    def process(self, image, scale=4):
        # Plugin logic here
        return upscaled_image

# Install plugin
flux plugins install real-esrgan

# Use in UI or API
result = flux.generate(..., plugins=["real-esrgan@4x"])
```

Browse the [Plugin Marketplace](https://plugins.flux.ai) for hundreds of ready-to-use plugins.

---

## 📊 Performance Benchmarks

| Operation | Stable Diffusion WebUI | **flux (TensorRT)** | Speedup |
|-----------|------------------------|---------------------|---------|
| SDXL 1024x1024 (30 steps) | 12.4s | **2.8s** | **4.4x** |
| Batch 4x (512x512) | 38.2s | **7.1s** | **5.4x** |
| LoRA switching | 3.2s | **0.4s** | **8.0x** |
| Model loading | 8.7s | **1.2s** | **7.3x** |

*Benchmarked on NVIDIA RTX 4090, CUDA 12.1*

---

## 🌟 Success Stories

> "We migrated 50 GPUs from A1111 to flux and cut our cloud costs by 60% while doubling output."  
> — **AI Startup, Series B**

> "The real-time collaboration feature let our design team work together like Google Docs for AI art."  
> — **Creative Agency, Fortune 500 Client**

> "From zero to production API in 15 minutes. The one-click deployment is game-changing."  
> — **Indie Developer**

---

## 🤝 Community & Support

- **Discord**: [Join 10,000+ users](https://discord.gg/flux)
- **GitHub Discussions**: [Ask questions](https://github.com/flux-ai/flux/discussions)
- **Documentation**: [docs.flux.ai](https://docs.flux.ai)
- **Enterprise Support**: [enterprise@flux.ai](mailto:enterprise@flux.ai)

### Contributing
We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### License
flux is [MIT licensed](LICENSE). Use it for anything.

---

## 🚦 Roadmap

- [ ] **Q4 2024**: Mobile app for iOS/Android
- [ ] **Q1 2025**: Video generation pipeline
- [ ] **Q2 2025**: Federated learning for custom models
- [ ] **Q3 2025**: 3D asset generation

---

**Ready to 10x your AI image generation?**  
[⭐ Star us on GitHub](https://github.com/flux-ai/flux) | [🚀 Deploy Now](https://deploy.flux.ai) | [💬 Join Discord](https://discord.gg/flux)

---

*flux is not affiliated with Stability AI or the original Stable Diffusion WebUI project. It is an independent, community-driven rewrite focused on performance and scalability.*