# Light Field Image Compression Codec (RKLT)

A C++ implementation of a hierarchical 4D encoder/decoder for light field image compression using transform coding and rate-distortion optimization.

## Overview

This project implements efficient compression of 4D light field images through:
- **Hierarchical 4D block partitioning** - Adaptive spatial/angular decomposition
- **Transform coding** - Spectral transformation of 4D blocks
- **Rate-distortion optimization** - Lagrange multiplier-based bit allocation
- **Probability modeling** - Entropy coding for compressed streams

## Building

```bash
cmake -B build
cmake --build build
```

Executables are generated in `bin/`:
- `MSGTEncoder` - Light field compression
- `MSGTDecoder` - Light field decompression

## Encoder Usage

```
MSGTEncoder [options]
```

### Required Options
- `-lf <directory>` - Input light field directory
- `-o <filename>` - Output compressed file
- `-nh <count>` - Number of horizontal views
- `-nv <count>` - Number of vertical views

### Block Configuration
- `-t, -s, -v, -u` - 4D block dimensions (t, s, v, u directions)
- `-min_t, -min_s, -min_v, -min_u` - Minimum partition sizes
- `-lambda <value>` - Lagrange multiplier for rate-distortion optimization

### Optional Parameters
- `-off_h, -off_v` - First view indices (default: 0,0)
- `-lenslet13x13` - Use lenslet light field format
- `-u_scale <gain>` - Arithmetic overflow control for large blocks
- `-cf <config_file>` - Load parameters from configuration file

## Decoder Usage

```
MSGTDecoder [options]
```

### Required Options
- `-lf <directory>` - Output light field directory
- `-i <filename>` - Input compressed file
- `-nh <count>` - Number of horizontal views
- `-nv <count>` - Number of vertical views

### Optional Parameters
- `-off_h, -off_v` - First view indices
- `-cf <config_file>` - Load parameters from configuration file

## Project Structure

- `src/Encoder/` - 4D hierarchical encoding with rate-distortion optimization
- `src/Decoder/` - Bitstream parsing and reconstruction
- `src/LightField/` - Light field data structures and 4D blocks
- `src/IO/` - File I/O and format handling
- `src/ProbabilityModel/` - Entropy coding models
- `01_TemplateFolder/` - Test configurations
- `results/` - Experiment outputs and evaluation metrics

