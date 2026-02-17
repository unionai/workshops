
# Flyte & Union.ai Tutorials

## Tutorial list

<table>
  <thead>
    <tr>
      <th align="left">Tutorial&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
      <th align="left">What you’ll learn</th>
      <th align="left">Open:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="tutorials/flyte2-quickstarts/00_flyte2-starter.ipynb">Get started with Flyte 2.0</a></td>
      <td>Get started with Flyte tasks, build an ML pipeline, handle errors, and run AI agents · <code>flyte.TaskEnvironment</code>, <code>flyte.ReusePolicy</code>, <code>flyte.map()</code></td>
      <td><a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/flyte2-quickstarts/00_flyte2-starter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a></td>
    </tr>
    <tr>
      <td><a href="tutorials/multi-agent-workflows/tutorial_planner_agent.ipynb">Planner Multi-Agent System</a></td>
      <td>create a scalable planner multi-agent system using Flyte</td>
      <td></td>
    </tr>
        <tr>
      <td><a href="tutorials/multi-agent-workflows/tutorial_react_agent.ipynb">ReAct Multi-Agent System</a></td>
      <td>create a scalable ReAct multi-agent system using Flyte</td>
      <td></td>
    </tr>
        <tr>
      <td><a href="tutorials/multi-agent-workflows/tutorial_reflection_agent.ipynb">Reflection Multi-Agent System</a></td>
      <td>create a scalable reflection multi-agent system using Flyte</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="tutorials/multi-agent-workflows/tutorial_debate_agent.ipynb">Debate Multi-Agent System</a></td>
      <td>create a scalable debate multi-agent system using Flyte</td>
      <td></td>
    </tr>
      <tr>
      <td><a href="tutorials/multi-agent-workflows/tutorial_manager_agent.ipynb">Manager Multi-Agent System</a></td>
      <td>create a scalable manager multi-agent system using Flyte</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="tutorials/multi-agent-workflows/tutorial_sequential_agent.ipynb">Sequential Multi-Agent System</a></td>
      <td>create a scalable sequential multi-agent system using Flyte</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="tutorials/mcp/tutorial_recipe_mcp.ipynb">Build a Recipe Assistant MCP Server</a></td>
      <td>build and deploy a Model Context Protocol (MCP) server on Union that helps AI agents find recipes using the Spoonacular Food API</td>
      <td></td>
    </tr>
  </tbody>
</table>



## Setup Instructions

You can clone repo: `git clone https://github.com/unionai/workshops`

It's reccomended to use UV and create a virtual environment at the `workshops` root directory. 

Then navigate to the tutorial directory for install requirments. 

```bash
# Clone the repository

# Create virtual environment
uv venv .venv --python 3.11

# Activate the venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r TUTORIALFOLDER/requirements.txt
```
