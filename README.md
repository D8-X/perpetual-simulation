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
    - One set of parameters is needed per perpetual and pool
- Output is written to ./results
  - One csv file per simulation, following the pattern
  `simulation_{Hash Code}.csv`
    containing a breakdown of the pool's revenue. This hash code identifies the entire set of simulation environments, not the parameters of each perpetual.
  - One csv file per perpetual, of the form 
  `results{Total Traders}{Total Arbs}_{Index}{Quote}{Collateral}-{End Date}-{Hash Code}.csv`
    containing a detailed view of each contract's state over time. The hash code identifies the individual simulation.
 

## How-to
- Open simulation.py
- The 'main' function contains the parameters defining the grid of scenarios to be simulated:
  - Random seed
  - Simulation period
  - Long/short probability
  - Initial investment in the pool
  - Average cash per trader
- Executing this script triggers a parallelized run of all the possible scenario configurations using these parameters.
- Global parameters are defined at the beginning of the script. These are shared across simulated scenarios, and include: 
  - Average trading frequency (based on observed activity in other exchanges)
  - External liquidity provision (number of stakers, amount per staker, holding periods)
  - Automated revenue withdrawal (for testing purposes, disabled)
  - Funding for arbitrage (for testing purposes, disabled)
- Partial results are periodically shown on screen.

## Components

### AMM

One per collateral currency, and so one per simulation. Corresponds to Liquidity Pools in the smart contracts. Implemented in amm.py.
### Perpetual

One per index, quote and collateral currency triple. Implemented in perpetual.py.
### Traders

Simulations consist of 3 different traders/agents:
#### Noise Traders
  - When they don't have an open position, they will stochastically try to open one.
  - When they have an open position, they will close using a Stop Loss/Take Profit logic. 
  - Individual preferences (how close to max leverage, SL/TP rates, slippage constraints) are randomized.
  - The cash available for each trader is randomized and based on the distribution observed in other exchanges.
#### Momentum Traders
  - They look for price movements that deviate from their moving average.
  - IWhen the deviation is large enough:
    - If they don't have an open position, they attempt to trade in the direction of the price movement.
    - If they have an open position, they will keep it open unless the direction goes in the opposite direction
  - When the deviation is not significant:
    - If they have an open position, they attemp to close it.
  - Individual preferences (deviation thresholds, slippage constraints) are randomized.
  - The cash available for each trader is randomized and based on the distribution observed in other exchanges.

#### Arbitrage Traders
  - They look for arbitrage opportunities between the exchange and one other centralized exchange.
  - Usually kept out of the simulations to make the calibrations more conservative: e.g. the mark premium rate in the presence of arbitrage should be at most as high as without arbitrage.

### Liquidity Providers
Correspond to external liquidity providers. Only "noise" LPs are implemented:
  - They deposit cash stochastically within a given period.
  - They withdraw their deposit after a given hoolding period has passed.
  - The cash available to each LP is randomized.

### Data-getter
Used to query index data directly from Chainlink.

### Analysis Scripts
Used to perform ad-hoc analysis of simulation results: pricing curve (analyse_price_function.py), arbitrage earnings (analysis_pnl.py), arbitrage earnings extrapolated to larger volume (analysis_pnlextrapolation.py), and overall liquidty and revenue observed per pertual and pool (analysis.py).

Of these, the most useful is analysis.py:
  - Executed based on one of the results_*.csv file produced by a simulation run.
  - We observe the premium rates (mark, mid, cex if present).
  - Trade sizes and slippage (max allowed position, typical traded position, minimal positions).
  - Liquidity pool status (per perpetual and aggregate, default fund, external liquidity, perpetual margin accounts).
  - Earnings from liquidations, liquidity provision, protocol withdrawals, and arbitrage.




