---
name: 🐛 Bug Report
about: Create a report to help us improve ASCOTA
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is. (e.g., "SAM failed to segment the sherd in low-light images.")

**Module Affected**
Select which part of the pipeline is failing:
- [ ] ascota_core (Preprocessing/Segmentation/Scale)
- [ ] ascota_classification (ViT/Deep Learning models)
- [ ] color_correct UI (Clustering/Calibration)
- [ ] preprocess UI
- [ ] classification UI
- [ ] sam_wand_ui / streamlit apps

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run command '...'
3. See error '...'

**Expected Behavior**
A clear and concise description of what you expected to happen.

**Screenshots / Data**
If applicable, add screenshots of the failed segmentation or the original archaeological image to help explain your problem. Additionally a copy of the .ascota file artifacts for the context. 

**Desktop/Environment:**
 - OS: [e.g. Windows 11, macOS]
 - Python Version: [e.g. 3.10]
 - GPU/CPU: [e.g. NVIDIA RTX 3060 or Apple M2]
 - CUDA version (if applicable)

**Additional Context**
Add any other context about the excavation data (e.g., specific sherd fabric or lighting conditions in the field).
