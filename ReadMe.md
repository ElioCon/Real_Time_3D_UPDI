Ultrasound Beamforming Framework - README

## Overview

This project provides a complete ultrasound beamforming framework supporting two distinct ultrasound systems:

1. **Verasonics Workflow** - For the Verasonics Vantage Research Ultrasound System using MATLAB acquisition scripts and Python/JAX beamforming
2. **X-phase Workflow** - For a custom FPGA-based ultrasound system with binary data processing

The framework supports both 2D and 3D imaging, coherent compounding, synthetic aperture processing, and Doppler flow imaging with SVD clutter filtering.

---

## System Requirements

### Hardware
- **Operating System**: Linux (required for JAX GPU support)
- **RAM**: 
  - Minimum 32 GB for 3D beamforming
  - Minimum 16 GB for 2D beamforming
- **GPU**: Optional but recommended for faster processing (CUDA-compatible)

### Software
- **Python**: Version 3.11 or higher
- **MATLAB**: Required for Verasonics data acquisition and processing (R2018b or later)
- **Verasonics Vantage Software**: Required for ultrasound data acquisition

### Python Dependencies
```bash
pip install jax jaxlib h5py numpy panel pillow matplotlib
```

For GPU acceleration (recommended):
```bash
pip install jax[cuda11_pip]  # For CUDA 11
# or
pip install jax[cuda12_pip]  # For CUDA 12
```

---

## Project Structure

```
Final_scripts/
│
├── Instructions.txt                    # Quick reference guide
│
├── Verasonics/                         # Verasonics System Workflow
│   ├── data/
│   │   ├── data_flow/                  # Flow/Doppler imaging data (simulation)
│   │   │   ├── 2D/                     # 2D probe flow data
│   │   │   └── 3D/                     # 3D probe flow data
│   │   └── data_static/                # Static phantom data
│   │       ├── 2D/                     # 2D probe static data
│   │       └── 3D/                     # 3D probe static data
│   │
│   ├── output/                         # Generated output files
│   │
│   └── scripts/
│       ├── matlab_acquisition/         # Verasonics acquisition scripts
│       │   ├── 2D/
│       │   │   ├── SetUpCM12_FlashAngles.m       # Original acquisition script
│       │   │   └── new_SetUpCM12_FlashAngles.m   # Updated acquisition (recommended)
│       │   └── 3D/                     # Placeholder for 3D probe scripts
│       │
│       ├── matlab_processing/          # MATLAB data processing utilities
│       │   ├── append_2D.m             # Append missing variables to .mat files
│       │   ├── fix_delay.m             # Fix TX delay mapping (3D probes)
│       │   ├── fix_probe_geometry.m    # Fix probe geometry mapping (3D probes)
│       │   └── save_raw.m              # Save Verasonics data to h5-compatible format
│       │
│       └── python_beamforming/         # Python beamforming modules
│           ├── beamformer.py           # Main beamforming script (saves images)
│           ├── define_delay.py         # Delay computation algorithms
│           ├── define_tof.py           # Time-of-flight beamforming (JAX)
│           ├── define_visualizer.py    # 3D visualization utilities
│           └── load_h5.py              # H5/MAT file loading utilities
│
├── X-phase/                            # X-phase FPGA System Workflow
│   ├── data/                           # Raw binary data storage
│   │   ├── RawData_2Frame/             # 2-frame acquisition
│   │   ├── RawData_36Frame/            # 36-frame acquisition
│   │   ├── RawData_7000pri_steer/      # Steered acquisitions
│   │   └── 2026-03-25_.../             # Additional acquisitions
│   │
│   ├── output/                         # Generated output files
│   │
│   └── scripts/
│       ├── FPGA*_n2Frame_16bit.bin     # Binary raw data (3 FPGA banks)
│       ├── test_x_phase.py             # Main test script
│       ├── test_x_phase2.py            # Alternative test script
│       ├── test_x_phase_small.py       # Simplified test script
│       │
│       └── python_beamforming/         # Python beamforming modules
│           ├── beamformer_images.py    # Image generation script
│           ├── beamformer_panel.py     # Interactive visualization script
│           ├── define_delay.py         # Delay computation
│           ├── define_tof.py           # Beamforming algorithms
│           ├── define_visualizer.py    # Visualization utilities
│           └── load_h5.py              # Data loading utilities
```

---

## Workflow Guides

### Verasonics Workflow

#### Step 1: Data Acquisition (MATLAB)

1. **Prepare Acquisition Script**
   - Use `new_SetUpCM12_FlashAngles.m` for 2D probe acquisitions
   - This script is configured for:
     - CM12 phased array probe
     - 7 compound angles (±7° span)
     - 500 frames
     - Sector imaging mode

2. **Run Acquisition**
   ```matlab
   % In MATLAB with Verasonics Vantage running:
   run('new_SetUpCM12_FlashAngles.m');
   % Acquisition will begin automatically
   ```

3. **Important**: Freeze VSX before saving data
   - Press freeze button in Verasonics interface
   - Data must be frozen to copy buffers

#### Step 2: Data Export (MATLAB)

After acquisition completes:

1. **Save Raw Data**
   ```matlab
   % Load your acquisition data in workspace
   % Then run save_raw with output path:
   save_raw('/path/to/output_file_name.mat')
   ```
   
   This creates an h5-compatible .mat file containing:
   - RcvData (raw channel data)
   - ProbeGeometry (element positions)
   - TxDelay (transmit delays)
   - PolarAngles (steering angles)
   - All necessary metadata

2. **For Legacy Data** (if missing parameters)
   ```matlab
   % If you have older .mat files missing required fields:
   % First load the file in workspace, then:
   append_2D;
   % Then save:
   save_raw('/path/to/output_file_name.mat')
   ```

#### Step 3: Beamforming (Python)

1. **Configure Input Path**
   ```python
   # Edit beamformer.py:
   input_path = './data_static/your_data_file.mat'
   ```

2. **Run Beamforming**
   ```bash
   cd Verasonics/scripts/python_beamforming/
   python beamformer.py
   ```

3. **Outputs** (saved to `./output_images/`):
   - `bmode_raw.npy` - Linear normalized B-mode image
   - `bmode_log_db.npy` - Log-compressed (dB) B-mode image
   - `bmode_linear.png` - PNG image (linear scale)
   - `bmode_log.png` - PNG image (dB scale, 60 dB dynamic range)

#### Optional: Interactive 3D Visualization

For 3D data visualization:
```python
# Uncomment visualization section in beamformer.py
# Or use beamformer_panel.py:
panel serve beamformer_panel.py
```

Features:
- VTK-based volume rendering
- Slice viewers (XY, XZ, YZ planes)
- Dynamic range control
- Frame navigation slider

---

### X-phase Workflow

#### Understanding the System

The X-phase system uses:
- **192-element linear probe** (pitch: 209 µm)
- **Center frequency**: 7 MHz
- **Sampling frequency**: 40 MHz
- **3 FPGA banks**: 64 channels each
- **Binary data format**: int16, column-major (Fortran order)

#### Step 1: Prepare Binary Data

Place your FPGA binary files in the scripts directory:
- `FPGA1_n2Frame_16bit.bin` (channels 0-63)
- `FPGA2_n2Frame_16bit.bin` (channels 64-127)
- `FPGA3_n2Frame_16bit.bin` (channels 128-191)

#### Step 2: Configure Parameters

Edit `test_x_phase.py` to match your acquisition:

```python
nCh = 64           # Channels per FPGA
nDepth = 1024      # Samples per channel
nPRI = 192         # PRIs per frame
nFrame = 2         # Number of frames
nPRI_tot = nPRI * nFrame  # Total PRIs
n_el = 192         # Total elements

c0 = 1540.0        # Speed of sound (m/s)
fc = 7e6           # Center frequency (Hz)
fs = 40e6          # Sampling frequency (Hz)
pitch = 209e-6     # Element pitch (m)
first_depth = 0.005  # First depth (m)
```

#### Step 3: Run Processing

```bash
cd X-phase/scripts/
python test_x_phase.py
```

#### Outputs:
- `beamformed_comparison.png` - Raw data vs beamformed image comparison
- `beamformed_detailed.png` - Single PRI vs averaged comparison
- Console output with timing information

---

## Module Documentation

### Core Beamforming Modules

#### `define_delay.py` - Delay Computation

**Functions:**

```python
create_delay_array_t0(grid, probe_geometry, t0_delays, polar_angles, 
                       c0, sampling_frequency, initial_time, pulse_peak_m)
```
- Computes receive delays using pre-calculated transmit delays from Verasonics
- Parameters:
  - `grid`: Pixel positions (meters), shape: (n_pix, 3)
  - `probe_geometry`: Element positions (meters), shape: (n_st, n_el, 3)
  - `t0_delays`: Transmit delays (meters), shape: (n_tx, n_st, n_el)
  - `polar_angles`: Steering angles, shape: (n_tx, 2)
- Returns: `delays` array, shape: (n_tx, n_st, n_pix, n_el)

```python
create_delay_array(grid, probe_geometry, polar_angles, c0, 
                   sampling_frequency, initial_time)
```
- Computes delays for plane wave imaging (without pre-calculated TX delays)
- Uses `distance_Tx_planewave_cpu()` for transmit distance calculation

#### `define_tof.py` - Beamforming Algorithms

**Core Beamforming:**

```python
beamform_pre_computed(data, delays)
```
- Applies delays to raw channel data
- Linear interpolation between adjacent samples
- Sum across elements (coherent summation)
- Parameters:
  - `data`: Raw data, shape: (n_ax, n_el)
  - `delays`: Delay array, shape: (n_pix, n_el)
- Returns: Beamformed image line, shape: (n_pix,)

**Coherent Compounding:**

```python
beamform_pre_computed_coherence_frames_lax_vmap(data_frame, delays_angle)
```
- Multi-frame coherent beamforming
- Uses JAX `lax.map` for efficient batch processing
- Parameters:
  - `data_frame`: Multi-frame data, shape: (N_frames, n_tx, n_st, n_ax, n_el)
  - `delays_angle`: Delays, shape: (n_tx, n_st, n_pix, n_el)
- Returns: Beamformed images, shape: (N_frames, n_pix)

**Doppler Processing:**

```python
doppler(n1, n2, data, delays)
```
- SVD-based clutter filtering for flow imaging
- Parameters:
  - `n1, n2`: SVD filter range (keep components n1 to n2)
  - `data`: Multi-frame data
  - `delays`: Pre-computed delays
- Returns: 
  - `data_img_doppler`: Doppler image
  - `data_img_Bmode_averaged`: Averaged B-mode
  - `data_imgs_Bmode`: Individual B-mode frames

**Hilbert Transform:**

```python
hilbert_transform(x, N=None, axis=-1)
```
- FFT-based analytic signal computation
- Used for envelope detection in B-mode imaging

#### `define_visualizer.py` - Visualization

**3D Volume Visualization:**

```python
create_visualizer(data)
```
- Creates interactive Panel dashboard with VTK volume rendering
- Features:
  - Frame slider navigation
  - Display mode selector (Raw / dB)
  - Dynamic range slider
  - Orthogonal slice viewers
  - Edge gradient control

**Image Processing:**

```python
process_img_python(img, frames, x_shape, y_shape, z_shape, z_start, z_end)
```
- Reshapes, crops, and normalizes beamformed images
- Converts to dB scale (20*log10)
- Returns: Normalized image and dB image

#### `load_h5.py` - Data Loading

**H5/MAT File Utilities:**

```python
dereference_index(file, dataset, index, event=None, subindex=None)
```
- Handles MATLAB's object reference system in h5 files
- Automatically dereferences or returns direct data

```python
decode_string(dataset)
```
- Decodes MATLAB string arrays from h5 format

---

## Configuration Options

### Grid Configuration

The beamforming grid is defined by parameters loaded from the .mat file:

```python
# From PData structure:
size_1, size_2, size_3       # Grid dimensions (pixels)
xOrg, yOrg, zOrg             # Grid origin (wavelengths)
pdelta_1, pdelta_2, pdelta_3 # Grid spacing (wavelengths)

# Grid generation:
X = np.arange(xOrg, xOrg + size_1 * pdelta_1, pdelta_1)
Y = np.arange(yOrg, yOrg - size_2 * pdelta_2, -pdelta_2)
Z = np.arange(zOrg, zOrg + size_3 * pdelta_3, pdelta_3)
```

### 2D vs 3D Detection

The system automatically detects imaging mode:

```python
# 2D mode: y_pdelta == 0
is_3d = pdelta[1] > 0

if not is_3d:
    z_shape = size_2  # Dimensions are swapped for 2D
    y_shape = 1
    yOrg = 0
```

### Doppler Processing Configuration

```python
# SVD filter settings:
n1 = 2      # First component to keep (filter low-frequency clutter)
n2 = N      # Last component (N = total frames)
```

Adjust `n1` based on clutter strength:
- Higher `n1`: More aggressive clutter rejection
- Lower `n1`: Preserve more signal (may include clutter)

---

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**

```
Solution: Reduce grid size or use chunked processing
- Decrease PData.Size parameters in acquisition script
- Process fewer frames at once
- Use GPU with sufficient VRAM
```

**2. Incorrect Image Depth**

```
Check pulse_peak_wl parameter in beamformer.py:
pulse_peak_wl = file["TW"]["peak"][0, 0] + 1
# The +1 offset is critical for matching Verasonics depth
```

**3. MATLAB h5py Compatibility**

```
Issue: MATLAB v7.3 format required
Solution: save_raw.m automatically saves in v7.3 format
- Do not use older .mat format (pre-v7.3)
```

**4. Probe Geometry Issues (3D Probes)**

```
Issue: Element mapping incorrect
Solution: Use fix_probe_geometry.m and fix_delay.m
- These scripts handle HVMux aperture mapping for 3D probes
- Element ordering differs from 2D probes
```

**5. JAX Not Using GPU**

```python
# Check devices:
import jax
print(jax.devices())

# Force CPU if needed:
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
```

### Performance Optimization

**JIT Compilation:**

All beamforming functions are JIT-compiled for performance:

```python
@partial(jax.jit, static_argnames=['n1', 'n2'])
def jax_compute_svd_jit(SIG, n1, n2):
    ...
```

First execution will be slower (compilation), subsequent runs are fast.

**Memory Management:**

```python
# Delete large arrays after use:
del raw_data
# Convert to JAX arrays efficiently:
data_jnp = jnp.array(data)  # One-time conversion
delays_jnp = jnp.array(delays)
```

---

## Notes for Future Development

### Recommended Improvements

1. **Acquisition Scripts**
   - Add 3D probe acquisition scripts (currently empty placeholder)
   - Implement diverging wave transmissions for wider field of view
   - Add automatic data saving after acquisition

2. **Beamforming Algorithms**
   - Implement adaptive beamforming (e.g., coherence factor weighting)
   - Add f-number apodization (partially implemented in `fnumber_mask()`)
   - Implement parallel beamforming for multi-threaded CPU processing

3. **Visualization**
   - Add movie export functionality
   - Implement ROI selection tools
   - Add measurement tools (distance, area)

4. **Data Management**
   - Implement automatic data compression
   - Add metadata logging (acquisition parameters, processing settings)
   - Create data validation checks