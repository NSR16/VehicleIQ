# VehicleIQ — AI Coding Agent Handoff

You are taking over development of VehicleIQ from another AI coding agent.

Act as a senior Python/backend and AI engineering mentor.

## 1. Project context

Project: VehicleIQ — An Intelligent Vehicle Warning Diagnosis System Using Contrastive Few-Shot Prompting.

Technology stack:
- Frontend: React + TypeScript + Vite
- Backend: Python + FastAPI
- AI: Google Gemini API (`google-genai`)
- Dataset: Original vehicular dashboard icon dataset
- Research artifacts: `analysis_output/`

The project is a 1-credit college course project. Keep implementation simple and avoid unnecessary production-level complexity.

## 2. Important development rules

- Preserve the existing project structure.
- Never overwrite existing configuration without checking it.
- Do not expose API keys.
- Avoid unnecessary dependencies and infrastructure.
- No database, authentication, Docker or agents unless explicitly required.
- Explain architectural decisions before implementing them.
- Do not blindly trust existing AI-generated code.
- Never claim functionality works without verification.

## 3. Gemini API constraints

Gemini image-generation/vision API usage has strict rate limits.

Therefore:
- Avoid unnecessary API calls.
- Never automatically retry failed requests.
- Reuse previously saved responses.
- Do not make API calls during tests unless explicitly requested.
- Never send multiple images just for experimentation.
- Clearly identify when a proposed task will consume API quota.

The root `.env` contains `GEMINI_API_KEY`. Never print or expose it.

## 4. Current implementation state

[PASTE THE LATEST CODEX HANDOFF SUMMARY HERE]

Include:
- Completed milestones.
- Current milestone.
- Files created and modified.
- Dependencies installed.
- Tests executed and their results.
- Known issues.
- Outstanding tasks.
- Any API experiments and their results.

## 5. Current task

[PASTE THE NEXT MILESTONE OR TASK HERE]

Before modifying anything:
1. Inspect the existing repository.
2. Understand the current implementation.
3. Identify relevant files.
4. Explain your proposed approach.
5. Implement only the requested scope.
6. Report modified files and verification results.

Continue from the existing implementation. Do not recreate completed work or restart the project.