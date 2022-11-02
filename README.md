# PerpetualSimulation
Simulation of perpetual swap DEX and traders

- the simulation is started with simulation.py  
- writes an output to ./results/*.csv
- use create_postgres.py to load the results into postgres database "amm_sim"

# Integration Testing
use integrationTests.py for integration testing test case

- Perpetual parameters and AMM fundings are in the script. 
- the S2-Oracle price is represented by idx_px which is printed to the console when running the script
- there are 3 traders and they trade according to ScheduleTraders.csv. -1 means they enter a short max position (given their cash), 1 the same long
- Script output says what should be happening when mocking the oracle and trading according to ScheduleTraders.csv (e.g., liquidations at a certain point)

# Rebalancing sequence
- liquidate
    - if ret(S2 & S3)>thresh:
        rebalance_perpetual (price move)
    - book trade (-> update K and params)
    - update AMM pool size target
    - distribute fees
    - rebalance perpetual
        - update_mark_price

- trade
    - if ret(S2 & S3)>thresh:
        rebalance_perpetual (price move)
    - book trade (-> update K and params)
    - update AMM pool size target
    - distribute fees
    - rebalance_perpetual
        - update_mark_price
