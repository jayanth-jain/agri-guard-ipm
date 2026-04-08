title: Agri-Guard
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
🌾 Agri-Guard: Precision Pest Management Environment
Agri-Guard is an OpenEnv-compliant reinforcement learning environment designed to evaluate AI agents in the context of sustainable agriculture. It simulates a 10x10 rice paddy in Andhra Pradesh, India, where agents must manage pest outbreaks under strict economic and biological constraints.

🚀 Motivation
India faces an annual crop loss of approximately $36 Billion due to pests. Smallholder farmers often operate with limited data and tight budgets, leading to over-reliance on chemicals. Agri-Guard models this "Information Gap," challenging agents to balance immediate chemical costs with long-term sustainable biological controls.

🛠 Action Space
The agent can perform one action per turn on a specific [x, y] coordinate:

Tool	Cost	Effect
Scout	$10	Reveals precise pest levels (reduces uncertainty).
Neem Oil	$2	Sustainable repellent; reduces pests by a small amount.
Chemical	$5	Highly effective unless the pest has developed resistance.
Biological	$15	Expensive but eliminates even chemical-resistant pests.
Abandon	$0	Sacrifices the cell (Health=0) to create a firebreak/quarantine.
🔍 Observation Space
The environment returns a structured observation:

Heatmap: A 10x10 integer array (0-9) representing the health of each cell.
Remaining Budget: The dollars left for the current task.
Sensor Data: Localized pest level readings at the center of the grid.
Message: Contextual feedback regarding task status and budget.
📋 Task Definitions
1. Point Outbreak (Easy)
Goal: Identify and treat a single infestation point on the 10x10 grid.
Challenge: Localize and treat the epicenter before pests spread radially.
Budget: $100.
2. Resource Dilemma (Medium)
Goal: Manage two simultaneous outbreaks at opposite corners.
Challenge: With a restricted $55 budget, the agent must use the abandon_cell action strategically to protect the core field.
Budget: $55.
3. Resistance Test (Hard)
Goal: Manage a chemical-resistant pest population.
Challenge: The agent must recognize treatment failures and pivot to biological controls.
Grading: Implements exponential reward decay to reward early diagnosis.
📈 Baseline Score
**point_outbreak: 0.72 | resource_dilemma: 0.45 | resistance_test: 0.31.
💻 Setup & Usage
Build: docker build -t agri-guard .
Run: docker run -p 7860:7860 agri-guard
Evaluate: python inference.py
🌐 Live Interactive API
The environment is live and can be tested directly via the Swagger UI: 👉 Click here to open the Interactive Dashboard :  https://monkjay-agri-guard-ipm.hf.space/docs
