# Claude Environment Instructions

## IMPORTANT RULES FOR CLAUDE

1. **ALWAYS Review Plans Before Implementation**
   - Anytime you plan something (code design, action plan, etc.), STOP and review for:
     - Correctness
     - Good design principles
     - Whether it reimplements existing functionality
   - State your review explicitly before proceeding

2. **NEVER Run the App**
   - Do NOT start the Concept MRI app or any other user applications
   - The user maintains control over app execution
   - Only provide code changes and explanations

3. **Wait for User Approval**
   - When creating todo lists or planning work, STOP and wait for user approval
   - Do NOT start implementation until the user explicitly tells you to proceed
   - The user is NOT in auto mode for a reason - respect their control

4. **Session Start Protocol**
   - At the beginning of every chat, read `ARCHITECTURE.yaml`
   - Read `CURRENTLY_WORKING_ON.md` to understand current context
   - Check for any `CLAUDE.md` updates

## Python Environment
This project uses a Windows-style virtual environment located at `venv311/`.

To run Python scripts in this project:
```bash
# Use the venv Python directly
./venv311/Scripts/python.exe script.py

# Or on Windows
venv311\Scripts\python.exe script.py
```

## API Keys and Configuration
API keys and sensitive configuration are stored in `local_config.py` (which is gitignored).

To use API keys in scripts:
```python
from local_config import OPENAI_KEY, XAI_API_KEY, GEMINI_API_KEY, MONGO_URI
```

## Testing LLM Features
To test LLM analysis features:
```bash
cd /mnt/c/Repos/ConceptualFragmentationInLLMsAnalysisAndVisualization
./venv311/Scripts/python.exe test_llm_comprehensive.py
```

## Common Issues
1. **Module not found**: Always use the venv Python: `./venv311/Scripts/python.exe`
2. **Encoding errors**: Use `encoding="utf-8"` when writing files with Unicode characters
3. **API keys**: Ensure `local_config.py` exists (copy from `local_config.py.example`)

## Project Structure
- Main library: `concept_fragmentation/`
- Web app: `concept_mri/`
- Experiments: `experiments/`
- Scripts: `scripts/`

## Current Focus
See `CURRENTLY_WORKING_ON.md` for the active task and recent changes.

## Key Design Principles
1. **No Reimplementation**: Always check if functionality exists before creating new code
2. **Single Source of Truth**: Use existing modules in `concept_fragmentation/`
3. **Clean Architecture**: Maintain separation between library, experiments, and UI
4. **Comprehensive Analysis**: When analyzing with LLMs, pass ALL data in single call for better pattern detection