# NeuroClean

A unified LLM-agent framework for automated EEG artifact removal, integrating traditional signal processing (wavelet transform, filtering, ICA) and deep learning (EORNet with Mamba SSM) under a single conversational interface.

> 📄 **Paper**: AAAI 2027 submission. See `../AAAI-Article-Template/denoise_agent_paper.tex`.

## Overview

EEG Denoise Agent provides **8 denoising skills** as tools that an LLM can autonomously select and orchestrate based on natural language queries. Users describe artifact symptoms (e.g., "remove eye blinks from frontal channels"), and the agent chooses the optimal method, configures parameters, runs denoising, and reports results — zero signal processing expertise required.

```
User Query: "Remove ocular artifacts from FP1-F7 in the first 10 seconds"
                    │
                    ▼
┌──────────────────────────────────────┐
│         LLM Reasoning Core            │
│  Artifact ID → Method Selection → Run │
└──────────────────────────────────────┘
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  ┌─────────┐ ┌─────────┐ ┌──────────┐
  │ Wavelet │ │   ICA   │ │  EORNet  │  ... 8 tools total
  └─────────┘ └─────────┘ └──────────┘
       │            │            │
       └────────────┼────────────┘
                    ▼
         Denoised EEG + Report
```

## Installation

```bash
# Core dependencies
pip install openai numpy scipy mne regex pyyaml

# Denoising method dependencies
pip install pywt                    # Wavelet denoising
pip install scikit-learn            # ICA denoising
pip install torch                   # EORNet (PyTorch)

# Optional: for full EORNet performance
pip install mamba-ssm               # Mamba SSM (requires CUDA)
```

## Quick Start

```python
from main import DenoiseAgent

agent = DenoiseAgent(
    config_path="config/config.json",
    file_name="path/to/your/eeg.edf",
    api_key="your-api-key",
    base_url="https://api.llm.ustc.edu.cn/v1",
)

result = agent.run("Remove 60 Hz powerline noise and eye blink artifacts from all channels.")
print(result["response"])
```

## Architecture

| Component | Description |
|-----------|-------------|
| `main.py` | `DenoiseAgent` class — loads EEG, builds system prompt, runs LLM-tool loop |
| `prompt.py` | System prompt with denoising method selection guidelines |
| `config/config.json` | Sampling rate, filter settings, prior knowledge |
| `tools/` | Auto-discovered tool modules (decorator-based registration) |
| `utils/` | Tool call parsing, message merging, formatting |
| `eval/` | Evaluation pipeline, agent demo, figure generation |

## Denoising Tools

| Tool | Method | Best For |
|------|--------|----------|
| `notchFilter` | IIR notch (Q=30) | 50/60 Hz powerline interference |
| `bandpassFilter` | Butterworth (order 4) | Low-frequency drift, high-frequency noise |
| `savitzkyGolayFilter` | S-G polynomial smoothing | Mild smoothing, waveform preservation |
| `adaptiveFilter` | LMS adaptive (order 32) | EOG/ECG removal with reference channel |
| `waveletDenoise` | DWT + 3 threshold modes (Visu/Sure/Bayes) | Motion artifacts, transients, non-stationary noise |
| `icaDenoise` | FastICA + auto artifact ID | Eye blinks, muscle artifacts (multi-channel) |
| `eornetDenoise` | Mamba SSM (depth=4, d_state=32) | Ocular artifacts (single-channel, deep learning) |
| `reflectData` | Raw signal extraction | Before/after comparison |

### Wavelet Denoising Details

- **Wavelet bases**: `db4`, `sym8`, `coif3`, `bior4.4`
- **Threshold methods**: soft, hard
- **Threshold selection**: VisuShrink (universal), SUREshrink (Stein's unbiased risk), BayesShrink (adaptive Bayesian)
- **Scaling factor**: adjustable α ∈ [0, 2]

### ICA Denoising Details

- **Algorithm**: FastICA (scikit-learn)
- **Artifact ID**: combined heuristic — kurtosis (30%) + low-frequency power < 5 Hz (40%) + spatial focality (30%)
- **Auto-removal**: top-k components exceeding threshold are zeroed and signal reconstructed

### EORNet Details

- **Architecture**: Conv1D embedding → 4× Mamba blocks → Conv1D output head
- **Input**: single-channel, 512 samples (2s @ 256 Hz)
- **Pretrained**: 10-fold CV on 3,400 EEG-EOG pairs, multi-SNR (−5 to +5 dB)
- **Fallback**: GRU-based model when `mamba_ssm` unavailable
- **Long signals**: overlap-add segmentation (50% overlap)

## Adding a New Denoising Tool

```python
# tools/my_denoiser.py
from tools import function_register

@function_register.register(
    name="myMethod",
    description="Description of the method...",
    parameters=[
        {"name": "name", "type": "List[str]", "description": "Channel names"},
        {"name": "start", "type": "int", "description": "Start time (s)"},
        {"name": "end",   "type": "int", "description": "End time (s)"},
    ],
    returns={"type": "object", "description": "Denoised signals per channel"}
)
def myMethod(name, start, end, config):
    from .registerData import getRegisteredData
    data = getRegisteredData()
    # ... your denoising logic ...
    return {"channel_0": denoised_signal.tolist()}
```

Save the file in `tools/` — it auto-registers via `pkgutil.walk_packages`.

## Evaluation

### Batch Evaluation (no API needed)

Compares all 7 denoising methods on synthetic EOG-contaminated EEG:

```bash
cd EEGDenoiseAgent
python -m eval.denoise_eval --n_samples 30 --snr_levels -5 -3 0 3 5 10
```

Outputs:
- `eval/denoise_report.txt` — formatted comparison report
- `eval/denoise_report.json` — detailed per-sample metrics

### Agent Demo (requires LLM API)

Tests the full LLM-agent pipeline on selected samples:

```bash
python -m eval.agent_demo --n_samples 2 --snr_levels 0 5
```

### Generate Paper Figures

```bash
python -m eval.generate_figure   # → ../AAAI-Article-Template/snr_breakdown.png
```

## Evaluation Results Summary

Evaluated on 180 EEG segments with EOG artifacts across 6 SNR levels (−5 to +10 dB):

| Method | RRMSE | Pearson r | SNR (dB) |
|--------|-------|-----------|----------|
| Bandpass (0.5–40 Hz) | 0.93 | 0.74 | 1.55 |
| ICA (kurtosis) | 0.96 | 0.74 | 1.67 |
| Notch (60 Hz) | 0.96 | 0.74 | 1.65 |
| Wavelet (sym8) | 0.97 | 0.71 | 1.46 |
| Wavelet (db4) | 0.97 | 0.71 | 1.43 |
| EORNet (GRU fallback) | 1.04 | −0.09 | −0.33 |
| Savitzky-Golay | 1.07 | 0.63 | 0.15 |

> **Note**: EORNet with `mamba_ssm` achieves RRMSE ≈ 0.19, Pearson r ≈ 0.97 (from EORNet paper). The GRU fallback is an untrained placeholder.

## Configuration Reference

```json
{
  "fs": 256,
  "l_freq": 0.5,
  "h_freq": 70,
  "notch_freq": 60,
  "segment_length": 512,
  "eornet_device": "cpu",
  "prior knowledge": {
    "freq_bands": {
      "delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 13],
      "beta":  [13, 30], "gamma": [30, 70]
    }
  }
}
```

## Project Structure

```
EEGDenoiseAgent/
├── main.py                       # DenoiseAgent entry point
├── prompt.py                     # System prompt construction
├── README.md                     # This file
├── config/
│   └── config.json               # Global configuration
├── tools/
│   ├── __init__.py               # Auto-discovery of tool modules
│   ├── register.py               # FunctionRegistry + @register decorator
│   ├── registerData.py           # Global EEG data singleton
│   ├── dataLoad.py               # EDF/FIF loading + bipolar montage
│   ├── preprocessing.py          # Filter + segment/reconstruct utils
│   ├── wavelet_denoise.py        # DWT-based denoising
│   ├── filter_denoise.py         # Bandpass, notch, LMS, Savitzky-Golay
│   ├── ica_denoise.py            # FastICA + auto artifact removal
│   ├── eornet_denoise.py         # EORNet deep learning denoising
│   ├── reflectData.py            # Raw signal extraction
│   └── localModels/
│       └── EORNet.py             # EORNet model (with GRU fallback)
├── utils/
│   ├── parseCalling.py           # <FUNCTION>/<ARGS> parser
│   ├── messageMerge.py           # Tool return → conversation
│   └── transFormat.py            # JSON formatting helper
├── eval/
│   ├── data_prep.py              # Synthetic noisy data generation
│   ├── denoise_eval.py           # Batch evaluation (no API)
│   ├── agent_demo.py             # Agent-driven demo (with API)
│   └── generate_figure.py        # Paper figure generation
└── results/
    └── EORNet_best.pth           # Pretrained EORNet weights (fold 0)
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.0 | LLM API client |
| `numpy` | ≥1.21 | Array operations |
| `scipy` | ≥1.7 | Signal processing, filtering |
| `mne` | ≥1.0 | EEG file I/O, preprocessing |
| `torch` | ≥2.0 | EORNet inference |
| `pywt` | ≥1.3 | Wavelet transform |
| `scikit-learn` | ≥1.0 | FastICA |
| `regex` | ≥2023 | Tool call parsing |
| `pyyaml` | ≥6.0 | YAML config |

## License

This project is released for research purposes. See the paper for citation information.

## Citation

```bibtex
@inproceedings{eegdenoiseagent2027,
  title   = {{EEG Denoise Agent}: A Unified {LLM}-Agent Framework for Automated {EEG} Artifact Removal},
  author  = {Anonymous},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year    = {2027}
}
```
