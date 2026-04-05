# ML4SCI OpenEnv Disk Analysis Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OpenEnv environment for autonomous agents to learn protoplanetary disk image analysis.**

Meta Scaler OpenEnv Hackathon Submission | Real-world ML4SCI Application

## 🚀 Quick Links

- **Live Environment**: [HuggingFace Space](https://huggingface.co/spaces/damsaychesse/disk-image-analysis)
- **Hackathon**: Meta Scaler OpenEnv
- **Domain**: Astronomy & Machine Learning for Science

---

## 📋 Overview

This project presents a complete **OpenEnv environment** where AI agents learn to analyze protoplanetary disk images through multi-task reinforcement learning. It automates astronomical image analysis tasks that currently consume 250+ hours monthly for astronomers.

### Key Features
- ✅ **3 Progressive Tasks**: Easy → Medium → Hard
- ✅ **Real-world Problem**: Based on ML4SCI EXXA research
- ✅ **Complete OpenEnv Implementation**: All required endpoints
- ✅ **Reward Shaping**: Task-specific grading logic
- ✅ **Live Deployment**: Running on HuggingFace Spaces
- ✅ **Production Ready**: Docker containerized

---

## 🎯 Tasks

### Task 1: Transit Detection (Easy) 🟢
Detect if a planet transit occurred in a light curve.
- **Input**: 32×32 disk image
- **Output**: Binary ("yes_transit" / "no_transit")
- **Reward**: 0.8 (correct) / 0.2 (wrong)

### Task 2: Disk Classification (Medium) 🟡
Classify disk type based on age and properties.
- **Input**: 32×32 disk image
- **Output**: 3-class ("young_disk" / "evolved_disk" / "debris_disk")
- **Reward**: 0.8 (correct) / 0.3 (partial credit)

### Task 3: Property Estimation (Hard) 🔴
Estimate disk physical properties (mass, inclination, scale height).
- **Input**: 32×32 disk image
- **Output**: Numeric value (0.0-1.0)
- **Reward**: Graduated by error margin (0.1 → 0.9)

---

## 🔧 Technical Architecture

### Environment (`src/environment.py`)
