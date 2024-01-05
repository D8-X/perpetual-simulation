import glob
import os
import re
import time
import requests
from web3 import Web3
import datetime
import numpy as np
import pandas as pd
import pytz
import bitmex
from pybit import HTTP
import matplotlib.pyplot as plt


MAX_NUM_CALLS = 100_000

MAX_NUM_OBS = 1_000_000

MAX_NUM_FAILS = 128

SAVE_EVERY = 1_000


def maybe_fetch_chainlink_data(
    prefix: str, # e.g. 'BTCUSD_BSC_Mainnet'
    node_url: str, # e.g. 'https://bsc-dataseed.binance.org/'
    token_address: str, # e.g. '0x264990fbd0A4796A3E3d8E37C4d5F87a3aCa5Ebf'
    fromdate : datetime.datetime=None, # defaults to the start of the present day
    todate : datetime.datetime=None, # defaults to the current time, but calls are still made so not very useful
    toroundID : int=None # more useful than todate, allows to skip calls
    ):

    # fromdate 
    if not fromdate:
        fromdate = datetime.datetime.now()
        fromdate = fromdate.replace(hour=0, minute=0, second=0, microsecond=0)
    # todate defaults to the current time
    if not todate:
        todate = datetime.datetime.now()
    
    # fetch oracle info from index
    web3 = Web3(Web3.HTTPProvider(node_url))
    abi = '[{"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"description","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint80","name":"_roundId","type":"uint80"}],"name":"getRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"version","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]'
    contract = web3.eth.contract(address=token_address, abi=abi)

    if toroundID is not None:
        print(f"toRoundID target: {toroundID}")
        roundId = toroundID
        is_good_roundID = False
        while not is_good_roundID:
            try:
                _, price, startedAt, timeStamp, answeredInRound = contract.functions.getRoundData(roundId).call()
                is_good_roundID = True
                print(f"round id {roundId}, date = {datetime.datetime.fromtimestamp(timeStamp)}, price = {price}")
            except:
                roundId -= 1
        print(f"toroundID was modified to {roundId}")
    else:
        latestData = contract.functions.latestRoundData().call()
        roundId = latestData[4]

    print(f"Initial from-date: {fromdate}")
    print(f"Initial to-date: {todate}")

    # check if there's already a file with this name
    os.makedirs("./data/index/chainlink", exist_ok=True)
    filename = f"./data/index/chainlink/{prefix}_{str(fromdate.date())}_{str(todate.date())}.csv"
    saved_files = glob.glob(f"./data/index/chainlink/{prefix}_*.csv")
    df = pd.DataFrame(columns=['roundId', 'timestamp', 'datetime', 'price'])
    for saved_file in saved_files:
        print(f"Found previously downloaded data in {saved_file}")
        df_saved = pd.read_csv(saved_file)[['roundId', 'timestamp', 'datetime', 'price']]
        print(f"containing {df_saved.shape[0]} observations, beginning {df_saved['datetime'].min()} and ending {df_saved['datetime'].max()}")
        df_saved['datetime'] = pd.to_datetime(df_saved['datetime'])
        df_saved['datetime'] = df_saved['datetime'].apply(lambda x: x.replace(tzinfo=pytz.utc))
        df = pd.concat([df, df_saved])
        df.drop_duplicates(subset='roundId', keep='first', inplace=True)
    df.sort_values('roundId', ascending=False, inplace=True)

    seen = set(df['roundId'].values)
    
    numObs = 0
    numCalls = 0
    numErrors = 0
    result = []
    tmp_filename = None
    dt = todate
    while numObs < MAX_NUM_OBS and numCalls < MAX_NUM_CALLS and numErrors < MAX_NUM_FAILS:
        # if observation already exists, nothing to do, except report once in a while
        if str(roundId) in seen: # df['roundId'].eq(str(roundId)).any():
            if numObs % 4096 == 0:
                mask = df['roundId'] == str(roundId)
                print(f"{numObs} {roundId} {df[mask]['datetime'].values} {df[mask]['price'].values}")
            roundId -= 1
            numObs += 1
            continue
        
        numCalls += 1
        try:
            if numCalls % 128 == 0:
                print(f"number of calls so far: {numCalls}, number of fetched and saved data points: {len(result)}/{numObs}")
            _, price, startedAt, timeStamp, answeredInRound = contract.functions.getRoundData(roundId).call()
            
            numObs += 1
            dt = datetime.datetime.fromtimestamp(timeStamp)
            
            if numObs % 32 == 0 or numObs == 0:
                # report queried observations once in a while
                print(f"{numObs} {roundId} {dt} {price / 100_000_000}")

            if dt < fromdate:
                # reached fromdate: exit loop
                numCalls = MAX_NUM_CALLS

            if dt <= todate and dt >= fromdate:
                # save data if timestamp is within required range
                dt = dt.replace(tzinfo=pytz.utc)
                result.append([roundId, timeStamp, dt, price / 100_000_000])
                # result.append([roundId, startedAt, answeredInRound, timeStamp, dt, price / 100_000_000])
        except Exception as e:
            numErrors += 1
            if numErrors % 25 == 0:
                print(e)
        
        if numErrors >= MAX_NUM_FAILS:
            print("Max number of failed calls reached.")

        if numCalls % SAVE_EVERY == 0 and len(result) > 0:
                # write to disk in case program is terminated so that calls are not wasted
                filename = f"./data/index/chainlink/{prefix}_{str(dt.date())}_{str(todate.date())}.csv"
                # put result list in output data frame and empty it
                df = pd.concat([df, pd.DataFrame(np.array(result), columns=['roundId', 'timestamp', 'datetime', 'price'])])
                # clean and save
                df.drop_duplicates(subset='roundId', keep='first', inplace=True)
                df.sort_values('timestamp', ascending=False, inplace=True)
                df.to_csv(filename)
                print(f"Progress saved to file {filename}")
                result = []
                if tmp_filename is not None and os.path.exists(tmp_filename) and tmp_filename != filename:
                    os.remove(tmp_filename)
                    print(f"Removed temporary file {tmp_filename}")
                tmp_filename = filename
                
        # roundId -= np.random.randint(1, 60)
        roundId -= 1

    fromdate = dt
    print(f"Effective from-date: {fromdate}")
    # combine
    filename = f"./data/index/chainlink/{prefix}_{str(fromdate.date())}_{str(todate.date())}.csv"
    if len(result) < 1 and tmp_filename is None:
        print("No new data was found in Chainlink")
        # return df
    elif len(result) > 0:
        df = pd.concat([df, pd.DataFrame(np.array(result), columns=['roundId', 'timestamp', 'datetime', 'price'])])
        df.drop_duplicates(subset='roundId', keep='first', inplace=True)
        df.sort_values('timestamp', ascending=False, inplace=True)
    # triple check dtypes
    df['price'] = pd.to_numeric(df['price'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=pytz.utc))
    # save
    df.to_csv(filename)
    print(f"Data succesfully saved to {filename}:")
    print(df.head(5))
    print(df.tail(5))
    return df


def fetch_bybit_data(bybit_idx):
    # TODO: these numbers look wrong, they don't match the bybit testnet FE (~10K vs real price)
    session = HTTP("https://api-testnet.bybit.com")
    todate = 1600544880
    fromdate = 1600543000
    resp = session.query_mark_price_kline(
        symbol="BTCUSD",
        interval=1,
        limit=200,
        from_time=todate
    )
    print(resp)

def fetch_bitmex_data(
    prefix: str, # e.g. 'BTCUSD_BSC_Mainnet'
    symbol: str, # e.g. '.BXBT'
    mainnet: bool, # True if mainnet data, False for testnet
    fromdate=None, 
    todate=None):
#     BITMEX_API_INFO = (
#     ('BTCUSD', 'BSC', 'Testnet', 'https://testnet.bitmex.com/api/v1/trade', '.BXBT'),
#     ('BTCUSD', 'BSC', 'Mainnet', 'https://www.bitmex.com/api/v1/trade', '.BXBT'),
# )
    client = bitmex.bitmex(test=not mainnet)
    #    datetime.datetime(2022, 3, 15, 19, 25, tzinfo=tzutc())
    if not todate:
        todate = datetime.date.today()
    todate_dt = pytz.UTC.localize(datetime.datetime(todate.year, todate.month, todate.day))
    if not fromdate:
        fromdate = datetime.date(2022, 1, 1)
    fromdate_dt = pytz.UTC.localize(datetime.datetime(fromdate.year, fromdate.month, fromdate.day))
    
    done = False
    result = []
    endTime = todate_dt
    numCalls = 0
    while not done:
        try:
            this_result = client.Trade.Trade_get(symbol=symbol, reverse=True, count=1000, endTime=endTime, startTime=fromdate_dt).result()
            numCalls += 1
            # for i in range(len(this_result[0]))
            if this_result[0][0]['timestamp'] <= todate_dt and this_result[0][-1]['timestamp'] >= fromdate_dt:
                result.extend(this_result[0])
            # end time shifts to earliest time
            endTime = this_result[0][-1]['timestamp']
            print(f"{this_result[0][-1]['timestamp']} ({this_result[0][-1]['price']})  --  {this_result[0][0]['timestamp']} ({this_result[0][0]['price']})")
            done = endTime <= fromdate_dt or numCalls > MAX_NUM_CALLS
        except:
            time.sleep(10)
    endTime -= datetime.timedelta(days=1)
    fromdate = datetime.date(endTime.year, endTime.month, endTime.day)
    os.makedirs("./data/index/bitmex", exist_ok=True)
    filename = f"./data/index/bitmex/{prefix}_{str(fromdate)}_{str(todate)}.csv"
    df = pd.DataFrame(result)
    df.to_csv(filename)
    print(f"Bitmex data successfully saved to {filename}")


def get_pyth_data(symbol, from_datetime, to_datetime):

    print(f"./data/index/pyth/{re.sub('/|^[a-zA-Z]+.', '', symbol)}")
    # https://hermes.pyth.network/api/get_price_feed?id=d6f83dfeaff95d596ddec26af2ee32f391c206a183b161b7980821860eeef2f5&publish_time=1704379736
    reqs = 0
    start_ts = time.time()
    from_ts = int((datetime.datetime.timestamp(from_datetime) // 60) * 60)
    to_ts = int((datetime.datetime.timestamp(to_datetime) // 60) * 60)
    assert(from_ts + 60 < to_ts)
    

    # df = pd.DataFrame(columns=['timestamp', 'datetime', 'price'])
    ts_range = np.arange(from_ts, to_ts, 86400)
    # results = np.zeros((ts_range.shape[0], 2))
    
    t = []
    p = []

    for ts in range(from_ts, to_ts, 86400):
        # endpoint = f"https://hermes.pyth.network/api/get_price_feed?id={id}&publish_time={ts_range[i]}"
        endpoint = f"https://benchmarks.pyth.network/v1/shims/tradingview/history?symbol={symbol}&resolution=1&from={ts}&to={ts+86400}"

        reqs += 1
        data = requests.get(endpoint).json()
        # print(data)

        if 'error' in data: # and data['error'] == 'too many requests':
            print(data['error'])
            # print("rate limit - waiting 60 seconds...")
            time.sleep(60)
            print("back")
            reqs = 0
        elif len(data['t']) > 0:
            price, publish_ts, start_price, start_publish_ts = data['c'][-1], data['t'][-1],  data['c'][0], data['t'][0]
            print(f"{datetime.datetime.fromtimestamp(start_publish_ts)}: {start_price}  ---  {datetime.datetime.fromtimestamp(publish_ts)}: {price}")
            # results[i,0] = publish_ts
            # results[i,1] = price
            t.extend(data['t'])
            p.extend(data['c'])
        else:
            print(f"No data: {datetime.datetime.fromtimestamp(ts)} - {datetime.datetime.fromtimestamp(ts+86400)}")
        
        # rate limit: 30 reqs in 10 secs
        if reqs >= 30:
            time.sleep(10)
            # ts_now = time.time()
            # wait_time = start_ts + 5 - ts_now if ts_now - start_ts < 5 else 1
            # time.sleep(wait_time)
            reqs = 0
            start_ts = time.time()
    # np.save(f"./{id}", results)
    df = pd.DataFrame({"timestamp": t, "price": p})
    df.to_csv(f"./data/index/pyth/{re.sub('/|^[a-zA-Z]+.', '', symbol)}_All_{datetime.datetime.fromtimestamp(t[0]).date()}_{datetime.datetime.fromtimestamp(t[-1]).date()}.csv")
    
def summarize_data(data, n_points=10_000, plot=True):
    # data.columns = roundId,timestamp,datetime,price
    # how many observations in a day?
    daily_obs_count = data.groupby(data['datetime'].dt.day)['price'].count()
    print("Daily data availability:")
    print(daily_obs_count.describe())

    # trigger analysis? TODO
    abs_ret = np.abs(np.diff(data['price'], prepend=np.nan) / data['price'])
    print("Tick-level returns statistics:")
    print(abs_ret.describe())
    print(np.nanquantile(abs_ret, (0.90, 0.99, 0.999, 0.9999)))
    plt.scatter(data['datetime'], abs_ret)
    plt.show()

    if not plot:
        return
    # we plot at most 200 data points so it doesn't look too cluttered
    for block in  [data.head(n_points), data.tail(n_points)]:
        df = block[::-1].copy()
        est = pytz.UTC # pytz.timezone('US/Eastern')
        df['hhmm'] = df['datetime'].map(lambda x: int(datetime.datetime.strftime(x.astimezone(est), '%H%M')))
        df['weekday'] = df['datetime'].map(lambda x: datetime.datetime.strftime(x.astimezone(est), '%A'))
        df['is_market_open'] = (df['weekday'] != "Saturday") & (df['weekday'] != "Sunday")# (df['hhmm'] >= 930) & (df['hhmm'] <= 1600) & (df['weekday'] != "Saturday") & (df['weekday'] != "Sunday")
        # df['is_market_open'] = df['datetime'].dt.weekday < 5
        
        plt.step(df['datetime'], df['price'], where='pre', marker='o')
        # gray on weekends
        i0 = 0
        while i0 < df.shape[0]:
            i = min([t for t in range(i0, df.shape[0]) if not df['is_market_open'].iat[t]], default=df.shape[0])
            j = min([t-1 for t in range(i + 1, df.shape[0]) if df['is_market_open'].iat[t]], default=df.shape[0]-1)
            if i < j:
                plt.axvspan(df["datetime"].iat[i], df["datetime"].iat[j], facecolor='0.2', alpha=0.1)
            i0 = j + 1
        plt.xticks(rotation=45, ha="right")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    # filename = "./data/index/chainlink/XAUUSD_BSC_Mainnet_2021-12-31_2022-09-08.csv"
    # filename = "./data/index/chainlink/BTCUSD_Polygon_Mainnet_2022-09-14_2022-09-16.csv"
    
    # df = pd.read_csv(filename)
    # df['datetime'] = pd.to_datetime(df['datetime'])
    # df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=pytz.utc))
    # print(df.head())
    # print(df.describe())
    # print(df['price'].describe())
    # abs_ret = np.abs(np.diff(df['price'], prepend=np.nan) / df['price'])
    # print("Tick-level returns statistics:")
    # print(abs_ret.describe())
    # print(np.nanquantile(abs_ret, (0.90, 0.99, 0.999, 0.9999)))
    # quit()
    

    # # BTC
    # df = maybe_fetch_chainlink_data(
    #     'BTCUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', 
    #     '0x264990fbd0A4796A3E3d8E37C4d5F87a3aCa5Ebf',
    #     fromdate=datetime.datetime(2022, 1, 1)
    # )
    # ETH
    # df = maybe_fetch_chainlink_data(
    #     'ETHUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', 
    #     '0x9ef1B8c0E4F7dc8bF5719Ea496883DC6401d5b2e',
    #     fromdate=datetime.datetime(2023, 1, 1)
    # )
    # # SOL
    # df = maybe_fetch_chainlink_data(
    #     'SOLUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', 
    #     '0x0E8a53DD9c13589df6382F13dA6B3Ec8F919B323',
    #     fromdate=datetime.datetime(2023, 1, 1)
    # )
    # # EUR
    # df = maybe_fetch_chainlink_data(
    # 'EURUSD_BSC_Mainnet', 
    # 'https://binance.nodereal.io', 
    # '0x0bf79F617988C472DcA68ff41eFe1338955b9A80',
    # fromdate=datetime.datetime(2023, 1, 1)
    # )
    # JPY
    # df = maybe_fetch_chainlink_data(
    # 'JPYUSD_BSC_Mainnet', 
    # 'https://binance.nodereal.io', 
    # '0x22Db8397a6E77E41471dE256a7803829fDC8bC57',
    # fromdate=datetime.datetime(2023, 1, 1)
    # )
    # TSLA
    # df = maybe_fetch_chainlink_data(
    #     'TSLAUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', 
    #     '0xEEA2ae9c074E87596A85ABE698B2Afebc9B57893',
    #     fromdate=datetime.datetime(2022, 4, 1)
    # )

    # # BNB
    # df = maybe_fetch_chainlink_data(
    #     'BNBUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io',#'https://bsc-dataseed.binance.org/', 
    #     '0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE',
    #     fromdate=datetime.datetime(2022, 4, 1)
    # )
    # Gold
    # df = maybe_fetch_chainlink_data(
    #     'XAUUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', #'https://bsc-dataseed.binance.org/', 
    #     '0x86896fEB19D8A607c3b11f2aF50A0f239Bd71CD0',
    #     fromdate=datetime.datetime(2022, 6, 1)
    # )
    # # Silver
    # df = maybe_fetch_chainlink_data(
    #     'XAGUSD_BSC_Mainnet', 
    #     'https://bsc-dataseed.binance.org/', 
    #     '0x817326922c909b16944817c207562B25C4dF16aD',
    #     fromdate=datetime.datetime(2023, 1, 1)
    # )
    # CHF
    # df = maybe_fetch_chainlink_data(
    #     'CHFUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', 
    #     '0x964261740356cB4aaD0C3D2003Ce808A4176a46d',
    #     fromdate=datetime.datetime(2023, 1, 1)
    # )
    # # SPY
    # df = maybe_fetch_chainlink_data(
    #     'SPYUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', 
    #     '0xb24D1DeE5F9a3f761D286B56d2bC44CE1D02DF7e',
    #     fromdate=datetime.datetime(2022, 1, 1)
    # )
    # 
    # GBP
    # df = maybe_fetch_chainlink_data(
    #     'GBPUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', 
    #     '0x8FAf16F710003E538189334541F5D4a391Da46a0',
    #     fromdate=datetime.datetime(2023, 1, 1)
    # )

    # # EUR on Polygon
    # df = maybe_fetch_chainlink_data(
    #     'EURUSD_Polygon_Mainnet', 
    #     'https://polygon-rpc.com/', 
    #     '0x73366Fe0AA0Ded304479862808e02506FE556a98',
    #     fromdate=datetime.datetime(2023, 1, 1)
    # )


    # BTC on Polygon
    # df = maybe_fetch_chainlink_data(
    #     'BTCUSD_Polygon_Mainnet', 
    #     'https://polygon-rpc.com/', 
    #     '0xc907E116054Ad103354f2D350FD2514433D57F6f',
    #     fromdate=datetime.datetime(2022, 9, 15)
    # )
    # # ETH on Polygon
    # df = maybe_fetch_chainlink_data(
    #     'ETHUSD_Polygon_Mainnet', 
    #     'https://polygon-rpc.com/', 
    #     '0xF9680D99D6C9589e2a93a78A04A279e509205945',
    #     fromdate=datetime.datetime(2023, 1, 1)
    # )
    # # CHF on Polygon # not slow
    # df = maybe_fetch_chainlink_data(
    #     'CHFUSD_Polygon_Mainnet', 
    #     'https://polygon-rpc.com/', 
    #     '0xc76f762CedF0F78a439727861628E0fdfE1e70c2',
    #     fromdate=datetime.datetime(2023, 1, 1)
    # )
    
    # # AVAX on Polygon: 0xe01eA2fbd8D76ee323FbEd03eB9a8625EC981A10
    # df = maybe_fetch_chainlink_data(
    #     'AVAXUSD_Polygon_Mainnet', 
    #     'https://polygon-rpc.com/', 
    #     '0xe01eA2fbd8D76ee323FbEd03eB9a8625EC981A10',
    #     fromdate=datetime.datetime(2022, 1, 1)
    # )
    
    # AVAX on Avalanche: 0x0A77230d17318075983913bC2145DB16C7366156
    # df = maybe_fetch_chainlink_data(
    #     'AVAXUSD_Avalanche_Mainnet', 
    #     'https://rpc.ankr.com/avalanche', 
    #     '0x0A77230d17318075983913bC2145DB16C7366156',
    #     fromdate=datetime.datetime(2022, 7, 1)
    # )
    
    # GBP on Polygon # slow
    # df = maybe_fetch_chainlink_data(
    #     'GBPUSD_Polygon_Mainnet', 
    #     'https://polygon-rpc.com/', 
    #     '0x099a2540848573e94fb1Ca0Fa420b00acbBc845a',
    #     fromdate=datetime.datetime(2023, 1, 12)
    # )
    
    # # Gold on Polygon
    # df = maybe_fetch_chainlink_data(
    #     'XAUUSD_Polygon_Mainnet', 
    #     'https://polygon-rpc.com/', 
    #     '0x0C466540B2ee1a31b441671eac0ca886e051E410',
    #     fromdate=datetime.datetime(2022, 1, 1)
    # )

    # MATIC on BSC
    # df = maybe_fetch_chainlink_data(
    #     'MATICUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', 
    #     '0x7CA57b0cA6367191c94C8914d7Df09A57655905f',
    #     fromdate=datetime.datetime(2023, 1, 1)
    # )

    # # LINK on BSC
    # df = maybe_fetch_chainlink_data(
    #     'LINKUSD_BSC_Mainnet', 
    #     'https://binance.nodereal.io', 
    #     '0xca236E327F629f9Fc2c30A4E95775EbF0B89fac8',
    #     fromdate=datetime.datetime(2022, 6, 1)
    # )
    # df = pd.read_csv("./data/index/chainlink/GBPUSD_Polygon_Mainnet_2023-01-11_2023-01-17.csv")[['roundId', 'timestamp', 'datetime', 'price']]
    # df['datetime'] = pd.to_datetime(df['datetime'])
    # df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=pytz.utc))
    # summarize_data(df, n_points=50_000)
    
    #  maybe_fetch_chainlink_data(4, fromdate=datetime.date(2021, 11, 1), toroundID=36893488147419285875) #, todate=datetime.date(2022, 3, 17))
    # Mainnet bitmex
    # fetch_bitmex_data(1, fromdate=datetime.date(2021, 1, 1), todate=datetime.date(2022, 3, 17))
    
    get_pyth_data("FX.GBP/USD", datetime.datetime(2023, 9, 1), datetime.datetime(2024, 1, 4))
        
    