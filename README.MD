# Tiny Recursive Model (TRM) - Local Code Agent Pipeline

This project provides a complete pipeline to train a **Tiny Recursive Model (TRM)** on your local codebase logic and deploy it as a VS Code agent. The goal is to create a model that understands the *reasoning* behind your code, not just the syntax.

## Prerequisites

- **Mac with Apple Silicon (M1/M2/M3)** recommended for MLX.
- **Python 3.10+**
- **Ollama** installed and running.
- **uv** (fast Python package manager) or standard `pip`.

## Phase 1: Setup Environment

1.  **Install uv** (Optional but recommended):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the MLX-TRM Repository**:
    We use the MLX implementation for Apple Silicon efficiency.
    ```bash
    git clone https://github.com/stockeh/mlx-trm.git trm-agent
    cd trm-agent
    ```

3.  **Install Dependencies**:
    ```bash
    uv sync
    # OR if using standard pip:
    # python -m venv .venv
    # source .venv/bin/activate
    # pip install -r requirements.txt
    ```

4.  **Install & Serve Teacher Model**:
    You need a larger model to generate reasoning data.
    ```bash
    # Install Ollama from https://ollama.com/
    ollama pull deepseek-r1:1.5b
    ```

## Phase 2: Create Context ("Logic Lab")

To verify the theory, we'll start with a small, logic-heavy workspace. You can add **any text-based code file** (Python, Java, JSON, XML, Rust, etc.).

1.  Create the workspace folder:
    ```bash
    mkdir -p my_workspace
    ```

2.  Add sample logic files. The generator will recursively scan this folder for *any* text files.
    - `factorial.py`: Recursive/iterative factorial.
    - `Sample.java`: Java class example.
    - `config.json`: JSON configuration.
    - `transform.xslt`: XML transformation.

## Phase 3: Generate Training Data

We "distill" reasoning from the teacher model (DeepSeek R1 1.5B) into a format TRM can learn (Input -> Reasoning -> Output).

1.  **Run the generator script**:
    ```bash
    python generate_data.py
    ```
    *This will recursively scan `my_workspace` for all text files (skipping binary/hidden files), ask Ollama to explain the logic step-by-step, and save the results to `data/custom_logic.jsonl`.*

## Phase 4: Train the Model

Train TRM to internalize the logic.

```bash
# Adjust batch_size based on your RAM (4-8 for 16GB RAM)
python train.py \
  --dataset custom \
  --data_path data/custom_logic.jsonl \
  --batch_size 4 \
  --layers 2 \
  --cycles 8 \
  --epochs 10
```

## Phase 5: VS Code Integration (MCP)

Expose your trained model to VS Code using the Model Context Protocol (MCP). This lets VS Code tools call a local server that exposes TRM-powered code reasoning.

1.  **Install FastMCP and MLX**:
    ```bash
    uv pip install fastmcp mlx
    ```

2.  **Run the Agent Server**:
    ```bash
    # From the project root
    python agent_server.py
    ```
    This starts a local MCP server that provides the `consult_logic_model` tool.

3.  **Install an MCP client in VS Code**:
    - Install an MCP-compatible extension (e.g., “MCP Inspector” or any MCP-enabled chat assistant).
    - These extensions let you register local MCP servers and call their tools from VS Code.

4.  **Configure the MCP server in VS Code**:
    Add a server entry to your MCP extension configuration (User Settings or the extension’s `mcp-servers.json`). Use absolute paths:

    ```json
    {
      "mcpServers": {
        "local-trm": {
          "command": "/Users/tamer.awad/workspaces/ai agents/CodeCoach/.venv/bin/python3",
          "args": ["/Users/tamer.awad/workspaces/ai agents/CodeCoach/agent_server.py"],
          "cwd": "/Users/tamer.awad/workspaces/ai agents/CodeCoach",
          "env": {
            "PYTHONPATH": "/Users/tamer.awad/workspaces/ai agents/CodeCoach/trm-agent"
          }
        }
      }
    }
    ```
    Notes:
    - `command` points to your virtualenv Python.
    - `args` launches the server script.
    - `PYTHONPATH` allows the server to import the TRM code when you later wire up model loading.

5.  **Use it in VS Code**:
    - Open your MCP-enabled chat panel.
    - Select the `local-trm` server.
    - Call the tool, for example: “Explain recursion in factorial.py” or “Trace the palindrome logic”.
    - The tool name is `consult_logic_model`; MCP clients typically let you invoke it directly or via the assistant UI.

---
**Files in this repo:**
- `generate_data.py`: Script to create dataset.
- `agent_server.py`: MCP server script.
- `my_workspace/`: Sample code for training.
