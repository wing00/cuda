# EECS 468, Winter 2017
Code for Northwestern's EECS 468 (Programming Massively Parallel Processors with CUDA). 

Contributors: Scott Young, Kapil Garg

## Labs
- Lab 1: Matrix Multiplication
- Lab 2: Tiled Matrix Multiplication
- Lab 3: Histograms
- Lab 4: Parallel Prefix Scan

## Development
1. Make sure the scaffold code is installed on a machine with a supported Nvidia GTX graphics card.
2. Clone this repository and replace the `labs/src` folder.
3. Run `source /usr/local/cuda-5.0/cuda-env.csh` or `. /usr/local/cuda-5.0/cuda-env.sh` (depending on environment) to setup the CUDA environment.
4. Run `make` from the scaffold's parent directory.
5. Run the compiled code as such `./labs/bin/linux/release/lab*` replacing the `*` with lab number to see results.
