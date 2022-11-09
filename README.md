# D8X Perpetual Simulations

## Overview

- Simulations are started by running simulation.py  
- Input is read from ./data/index and ./data/params
  - ./data/index consists of historical market data for each perpetual index and collateral
    - Time series is collected directly from Chainlink
    - BSC is used as source network in most cases
    - Some series, e.g. CHF, are taken from Polygon
    - The choice of source is roughly based on which network provides minute-frequency data
  - ./data/params contains the calibrated parameters for each perpetual contract
    - Each set of parameters is stored in a json file
- Output is written to ./results/*.csv
  - One csv file per perpetual, containing a detailed view of each contract's state over time
  - One csv file per simulation, containing a breakdown of the pool's revenue

## How-to
- Open simulation.py
- The first function in the script, 'main', contains all the crucial simulation parameters:
  - Random seed
  - Simulation period
  - Long/short probability
  - Initial investment in the pool
  - Average cash per trader
- Other parameters are shared across simulated scenarios and defined later in main: they may be modified but not in bulk, they are static for a given set of simulations
- 
