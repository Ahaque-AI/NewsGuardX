# NewsGuardX: The AI-Powered News Filtering Crewai Comprehensive Workflow

NewsGuardX is a multi-agent AI system built on the powerful [crewAI](https://crewai.com) framework. This project harnesses the collective intelligence of specialized agents to streamline complex workflows—ideal for tasks like news verification, content moderation, and ensuring information integrity.

> **Note:** This project template leverages crewAI's CausalityNet framework, empowering your agents to collaborate and execute defined tasks efficiently. Tailor your agents, tasks, and logic to meet your unique operational requirements.

## Overview

**NewsGuardX** is designed to leverage state-of-the-art AI technologies to monitor, analyze, and manage news content flows. By coordinating multiple specialized agents, the system can:

- **Filter and Verify News Sources:** Employ AI-powered agents to scrutinize news feeds.
- **Collaborate on Complex Tasks:** Facilitate agent teamwork to verify claims and flag misinformation.
- **Adapt Dynamically:** Easily adjust roles and logic to address emerging challenges.

This solution is perfect for automating research, streamlining reporting, and implementing robust news verification workflows.

## Features

- **Multi-Agent Collaboration:** Divide complex workflows among specialized AI agents.
- **Easy Configuration:** Edit YAML files to adjust agent roles, tasks, and settings effortlessly.
- **Seamless Dependency Management:** Leverage [UV](https://docs.astral.sh/uv/) for efficient package handling.
- **Customizable Workflow:** Define and update agent tasks to adapt to different news analysis scenarios.
- **Rapid Prototyping:** Quickly deploy and iterate using the provided commands and templates.

## Installation

### Prerequisites

- **Python:** Ensure Python version >=3.10 and <3.13 is installed. (Recommended is python=3.10.0)
- **UV Package Manager:** Used for dependency management and package handling.

### Steps

1. **Install UV:**  
   ```bash
   pip install uv
   ```

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/Ahaque-AI/NewsGuardX.git
   cd NewsGuardX
   ```

3. **Install Project Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create or update the `.env` file with your API keys:
   
   You will need 9 Environment Variables:
   - `GEMINI_API_KEY`
   - `SERPER_API_KEY`
   - `GROQ_API_KEY`
   - `TAVILY_API_KEY`
   - `NVIDIA_NIM_API_KEY`
   - `NEO4J_URI`
   - `NEO4J_USERNAME`
   - `NEO4J_PASSWORD`
   - `AURA_INSTANCEID`

5. **Define Agents:**
   Edit `src/causalitynet/config/agents.yaml` to set up your agent profiles, specifying roles, tools, and parameters.

6. **Define Tasks:**
   Modify `src/causalitynet/config/tasks.yaml` to outline tasks that the system should execute.

7. **Customize Workflow:**
   Update:
   - `src/causalitynet/crew.py` for custom logic and tool integrations.
   - `src/causalitynet/main.py` to manage custom inputs and runtime arguments.

8. **Usage:**
   ```bash
   crewai run
   ```

## Project Structure

```
NewsGuardX/
│
├── src/
│   └── causalitynet/
│       ├── config/
│       │   ├── agents.yaml    # Define agent parameters and tools
│       │   └── tasks.yaml     # List tasks for agent execution
│       ├── crew.py            # Main logic for agent collaboration
│       └── main.py            # Entry point for custom input and execution
│
├── .env                     # Environment variables (API keys, etc.)
├── crewai_flow.html         # HTML visualization of the workflow (optional)
├── docker-compose.yaml      # Docker configuration for containerized execution
├── dockerfile               # Dockerfile for building the project image
├── pyproject.toml           # Python project configuration
├── requirements.txt         # Additional project dependencies
└── uv.lock                  # UV dependency lock file
```

## Contributing

We welcome your contributions to enhance NewsGuardX! To contribute:

1. Fork the repository.
2. Create a new branch with your feature or fix.
3. Submit a pull request detailing your changes.

For major contributions, please open an issue first to discuss your ideas.

## Support

For support, questions, or feedback:

### Documentation
Visit the [crewAI Documentation](https://docs.crewai.com/introduction) for detailed guides and examples.

### GitHub Issues
Report bugs or request features via the [NewsGuardX GitHub Issues](https://github.com/Ahaque-AI/NewsGuardX/issues).
