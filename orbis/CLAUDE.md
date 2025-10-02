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
# Local development with optional local embeddings
uv sync --extra local

# Azure-only minimal dependencies
uv sync
```

### Folders to ignore
Always exclude .venv and __pycache__ folders when analysing folders.

## High-Level Architecture

Refer to docs/AGENTIC_RAG_SYSTEM_DIAGRAM.md and docs/AGENTIC_RAG_IMPLEMENTATION.md

## Implementation Lessons Learned

Critical success factors identified from the project's evolution include: adopting configuration-driven architecture over hardcoded implementations to achieve true modularity (reducing new data source integration from 20+ files to 1 connector + 1 config entry), implementing complete decoupling through generic services that eliminate system-wide assumptions while maintaining sophisticated multi-agent functionality, leveraging multi-stage agentic RAG workflows with project-aware intelligence and multi-source search for enhanced ticket analysis, and embedding configuration systems to optimize semantic search quality. Core principle: KISS. When implementing fixes, always distinguish between band-aid solutions and root cause fixes, think about whether you are addressing the underlying issue and not the symptom. Avoid creating technical debt and inconsistencies.

## Documentation Standards

Documentation (.md) should be direct and focused. Avoid excessive bullet points unless they add clear value. Minimize tangential topics like security or performance unless directly relevant. Keep each document narrowly scoped to prevent overlap with other files, making updates easier to track. Minimize usage of emojis.