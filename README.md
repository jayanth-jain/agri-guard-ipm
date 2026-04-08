---
title: Agri-Guard
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
---

# 🌾 Agri-Guard: Precision Pest Management Environment

**Agri-Guard** is an OpenEnv-compliant reinforcement learning environment that simulates a **10×10 rice paddy field in Andhra Pradesh, India**, where AI agents must manage pest outbreaks under **budget and sustainability constraints**.

---

## 🚀 Why Agri-Guard?

✔ Models real-world **$36B crop loss problem in India**  
✔ Simulates **low-information farming environments**  
✔ Encourages **sustainable decision-making**  
✔ Balances **cost vs long-term ecological impact**  

---

## 🛠 Action Space

Each turn, the agent performs **one action on `[x, y]`**:

| ✔ Tool        | Cost | Effect |
|--------------|------|--------|
| Scout        | $10  | Reveals exact pest level |
| Neem Oil     | $2   | Eco-friendly, small reduction |
| Chemical     | $5   | Strong effect (fails if resistant) |
| Biological   | $15  | Works even on resistant pests |
| Abandon      | $0   | Sacrifice cell to stop spread |

---

## 🔍 Observation Space

✔ **Heatmap** → 10×10 grid (0–9 health levels)  
✔ **Remaining Budget** → Current funds  
✔ **Sensor Data** → Local pest readings  
✔ **Message** → Status + feedback  

---

## 📋 Tasks

### 🟢 Point Outbreak (Easy)
✔ Detect single infestation  
✔ Stop radial spread  
💰 Budget: $100  

---

### 🟡 Resource Dilemma (Medium)
✔ Handle 2 outbreaks  
✔ Use abandon strategically  
💰 Budget: $55  

---

### 🔴 Resistance Test (Hard)
✔ Detect chemical failure  
✔ Switch to biological control  
✔ Faster response = higher reward  

---

## 📈 Baseline Performance

✔ Point Outbreak → **0.72**  
✔ Resource Dilemma → **0.45**  
✔ Resistance Test → **0.31**  

---

## 💻 Setup, Run & Evaluate (All-in-One)

### 1️⃣ Build the Environment
## ⚡ Quick Start (Setup + Run + Evaluate)

Run everything in one go:

```bash
docker build -t agri-guard . && \
docker run -d -p 7860:7860 agri-guard && \
sleep 5 && \
python inference.py
```

🌐 Live Interactive API
The environment is live and can be tested via Swagger UI.

👉 Open Interactive Dashboard : https://monkjay-agri-guard-ipm.hf.space/docs
