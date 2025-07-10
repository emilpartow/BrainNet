# DynamicMaskNet

**Biologically Inspired Dynamic Connectivity Neural Network Demo**

DynamicMaskNet is a minimal PyTorch experiment simulating dynamic, brain-inspired connectivity in artificial neural networks.  
On every forward pass, active connections (weights) are chosen randomly, mimicking how real neural circuits change their effective connectivity depending on context, input, or noise.

## Features

- Dynamic masking in each layer (random connectivity per forward pass)
- PyTorch implementation, easily extendable
- Visualizes masks and activation patterns
- Compare with standard dense (fully connected) layers

## Project Structure

DynamicMaskNet/
│
├── README.md
├── dynamic_mask_net.py # Main network code
├── visualize.py # Visualization tools
├── example_run.py # Demo and example run
├── requirements.txt
└── data/ # (Optional) for sample data

## Quickstart

### 1. Install requirements:
```sh
pip install -r requirements.txt
```

### 2. Run the example:
```py
python example_run.py
```


## Example Output

- Every run uses a new random connectivity mask.
- Masks and output activations are displayed for each run.

---

**Feel free to extend this project, use structured or learned masks, or compare to classic networks!**

