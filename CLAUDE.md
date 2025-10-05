# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Application
```bash
# Always activate virtual environment first
source .venv/bin/activate  # Bash
.\.venv\Scripts\Activate.ps1  # PowerShell

# Start the API server
uv run main.py
# Server runs on http://localhost:7887 by default
```

### Dependencies
```bash
# Install all dependencies
uv sync
```

### Folders to ignore
Always exclude .venv and __pycache__ folders when analysing folders.

## Monorepo approach

The repo contains currently two application:
- Orbis, a powerful agentic RAG capable of ingesting heterogeneous data sources, and performing powerful analyzes to provide detailed technical answer.
- Orbis-search, a search system for Azure DevOps ticket, using ensemble search.

In addition, the repo contains Orbis-core, a library with components common to both applications. The purpose of Orbis-core is not to hold code for both, it is strictly to share common components that are not customized within orbis-core.

## Implementation Lessons Learned

Critical success factors identified from the project's evolution include: adopting configuration-driven architecture over hardcoded implementations to achieve true modularity, core principle: KISS. When implementing fixes, always distinguish between band-aid solutions and root cause fixes, think about whether you are addressing the underlying issue and not the symptom. Avoid creating technical debt and inconsistencies.

## Documentation Standards

Documentation (.md) should be direct and focused. Avoid excessive bullet points unless they add clear value. Minimize tangential topics like security or performance unless directly relevant. Keep each document narrowly scoped to prevent overlap with other files, making updates easier to track. Minimize usage of emojis.