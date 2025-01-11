import requests
from config import OANDA_API_KEY, OANDA_ACCOUNT_ID, BASE_URL, TRADE_INSTRUMENT
from ml_model import MLModel
import logging
import numpy as np
import os
import json
import sqlite3
import time
import threading
import talib
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.model_selection import train_test_split

# Ensure logs folder
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

class TradingBot:
    def __init__(self, ml_model=None):
        # Enhanced logging setup with file and console handlers
        self.logger = logging.getLogger('TradingBot')  # Changed from local 'logger' to 'self.logger'
        self.logger.setLevel(logging.INFO)

        # File handler for persistent logging
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'trading_bot.log'))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self.logger.info("Initializing TradingBot...")

        # Credentials and configuration
        self.api_key = OANDA_API_KEY
        self.account_id = OANDA_ACCOUNT_ID
        self.ml_model = ml_model if ml_model else MLModel()

        # Basic thresholds and retraining scheduling
        self.retrain_threshold = 1000
        self.retry_count = 3
        self.scheduler = BackgroundScheduler()
        self.training_interval_hours = 24
        self.scheduler.add_job(self.run_job_queue, 'interval', minutes=5)
        self.scheduler.add_job(self.retrain_model, 'interval', hours=self.training_interval_hours)
        self.scheduler.start()

        # Initialize bot settings with default values
        self.risk_level = "medium"
        self.max_trades = 5
        self.trade_interval_hours = 24
        self.stop_loss_percentage = 2.0  # Ensure this is initialized
        self.take_profit_percentage = 5.0  # Ensure this is initialized
        self.trailing_atr_multiplier = 1.5
        self.adjust_atr_multiplier = 1.5
        self.trade_cooldown = 60
        self.position_units = 1000  # Example units; adjust as needed

        # API headers and initial load
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.account_balance = 0.0
        self.initial_balance = 0.0
        self.load_account_balance()
        self.initial_balance = self.account_balance

        # Trade & metrics state
        self.trades = []
        self.trade_count = 0
        self.profit_loss = 0.0
        self.success_rate = 0.0
        self.new_data_count = 0
        self.historical_metrics_file = 'historical_metrics.json'

        # Position tracking (FIFO)
        self.open_position = False
        self.current_side = None
        self.entry_price = None
        self.take_profit = None
        self.stop_loss = None
        self.position_units = 1000

        # Bot run state
        self.running = False
        self.last_trade_time = None
        self.trade_cooldown = 60
        self.start_time = None

        # Performance/accuracy tracking
        self.accuracy_history = []
        self.performance_threshold = 0.6
        self.positions = []  # For partial closes
        self.last_candle_time = None

        # SL/TP Adjustment Settings (Default Values)
        self.trailing_atr_multiplier = 1.5  # Multiplier for ATR in trailing stop
        self.adjust_atr_multiplier = 1.5    # Multiplier for ATR in take profit/stop loss adjustment

        self.logger.info("TradingBot initialized successfully.")
        self.setup_logging()
        self.load_state()
        self.load_historical_metrics()

        # Initial training if scaler and PCA are not fitted
        if not self.ml_model.is_ready():
            self.logger.info("[TradingBot] MLModel not ready. Performing initial training.")
            initial_price_data = self.get_prices(count=500, timeframe="H1")
            if initial_price_data:
                X_train, y_train, X_val, y_val = self.prepare_training_data(initial_price_data)
                if X_train.size > 0 and y_train.size > 0 and X_val.size > 0 and y_val.size > 0:
                    self.ml_model.train(X_train, y_train, X_val, y_val)  # Now passing four arguments
                    self.logger.info("[TradingBot] Initial training completed successfully.")
                else:
                    self.logger.error("[TradingBot] Insufficient data for initial training.")
            else:
                self.logger.error("[TradingBot] Failed to fetch price data for initial training.")

    def setup_logging(self):
        """Additional logging setup if needed."""
        pass  # Placeholder for any additional logging setup

    def update_training_interval(self, hrs):
        self.training_interval_hours = hrs
        for job in self.scheduler.get_jobs():
            if job.func == self.retrain_model:
                job.reschedule(trigger='interval', hours=hrs)
                self.logger.info(f"Training interval updated to {hrs} hours.")
                break

    def load_account_balance(self):
        """Load current account balance from OANDA."""
        for _ in range(self.retry_count):
            try:
                url = f"{BASE_URL}/accounts/{self.account_id}/summary"
                self.logger.info(f"[load_account_balance] Fetching account summary from {url}")
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                self.account_balance = float(data['account']['balance'])
                self.logger.info(f"[load_account_balance] Account balance: {self.account_balance}")
                return
            except Exception as e:
                self.logger.error(f"[load_account_balance] Error: {e}")
                time.sleep(1)

    def load_historical_metrics(self):
        """
        Load historical trade counts and profit/loss from the database.
        """
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    trade_count INTEGER,
                    profit_loss REAL
                )
            ''')
            cursor.execute('SELECT trade_count, profit_loss FROM metrics WHERE id=1')
            row = cursor.fetchone()
            if row:
                self.trade_count, self.profit_loss = row
                self.logger.info(f"[load_historical_metrics] Loaded trade_count={self.trade_count}, profit_loss={self.profit_loss}")
            else:
                cursor.execute('INSERT INTO metrics (id, trade_count, profit_loss) VALUES (1, 0, 0.0)')
                conn.commit()
                self.logger.info("[load_historical_metrics] Initialized metrics in database.")
            conn.close()
        except Exception as e:
            self.logger.error(f"[load_historical_metrics] Failed to load metrics: {e}")

    def save_historical_metrics(self):
        """
        Save current trade counts and profit/loss to the database.
        """
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    trade_count INTEGER,
                    profit_loss REAL
                )
            ''')
            cursor.execute('''
                UPDATE metrics SET trade_count=?, profit_loss=? WHERE id=1
            ''', (self.trade_count, self.profit_loss))
            conn.commit()
            conn.close()
            self.logger.info(f"[save_historical_metrics] Saved trade_count={self.trade_count}, profit_loss={self.profit_loss}")
        except Exception as e:
            self.logger.error(f"[save_historical_metrics] Failed to save metrics: {e}")

    def synchronize_positions(self):
        """Sync local positions with OANDA's open trades."""
        oanda_trades = self.get_open_trades()
        if not oanda_trades:
            self.logger.info("[synchronize_positions] No open trades on OANDA to synchronize.")
            self.positions = []
            return

        self.positions = [
            {
                "id": t.get("id"),
                "side": "buy" if float(t.get("currentUnits", 0)) > 0 else "sell",
                "units": abs(int(float(t.get("currentUnits", 0)))),
                "entry_price": float(t.get("price", 0)),
                "trade_id": t.get("tradeID"),  # Ensure this field exists
                "sl_order_id": t.get("sl_order_id"),  # Custom field, manage as needed
                "tp_order_id": t.get("tp_order_id")   # Custom field, manage as needed
            }
            for t in oanda_trades
        ]
        self.logger.info(f"[synchronize_positions] Synchronized {len(self.positions)} positions with OANDA.")

    def multi_timeframe_analysis(self):
        """Analyze data across multiple timeframes for more reliable signals."""
        lower_tf = self.get_prices(count=100, timeframe="M15")
        higher_tf = self.get_prices(count=100, timeframe="H4")

        if not lower_tf or not higher_tf:
            self.logger.error("[multi_timeframe_analysis] Failed to fetch multi-timeframe price data.")
            return 0, 0

        lower_indicators = self.evaluate_indicators(lower_tf)
        higher_indicators = self.evaluate_indicators(higher_tf)

        buy_signals = lower_indicators["summary"]["buy_signals"] + higher_indicators["summary"]["buy_signals"]
        sell_signals = lower_indicators["summary"]["sell_signals"] + higher_indicators["summary"]["sell_signals"]

        self.logger.info(f"[multi_timeframe_analysis] Combined Buy: {buy_signals}, Combined Sell: {sell_signals}")
        return buy_signals, sell_signals

    def analyze_open_trades(self):
        """Analyze existing open trades to decide if we hold, exit, or adjust."""
        self.synchronize_positions()
        if not self.positions:
            self.logger.info("[analyze_open_trades] No open positions to analyze.")
            return

        current_price = self.get_current_price()
        if not current_price:
            self.logger.error("[analyze_open_trades] Failed to fetch current price.")
            return

        price_data = self.get_prices(count=500)
        if not price_data:
            self.logger.error("[analyze_open_trades] Failed to fetch price data.")
            return

        # Evaluate indicators (dict-based)
        indicators = self.evaluate_indicators(price_data)
        if not indicators:
            self.logger.error("[analyze_open_trades] Failed to evaluate indicators.")
            return

        for pos in self.positions:
            side = pos['side']
            entry_price = pos['entry_price']
            units = pos['units']
            decision = "hold"

            # Compute P/L
            if side == "buy":
                profit_loss = (current_price - entry_price) * units
            else:
                profit_loss = (entry_price - current_price) * units

            # Basic logic to close if RSI or Bollinger triggers overbought/oversold
            if side == "buy":
                if (indicators["indicators"]["rsi"] > 70 or current_price >= indicators["indicators"]["boll_upper"]):
                    decision = "close"
            else:  # "sell"
                if (indicators["indicators"]["rsi"] < 30 or current_price <= indicators["indicators"]["boll_lower"]):
                    decision = "close"

            self.logger.info(
                f"[analyze_open_trades] Trade ID: {pos.get('id', 'N/A')} | "
                f"Side: {side.upper()} | Entry: {entry_price:.5f} | Units: {units} | "
                f"Current Price: {current_price:.5f} | P/L: {profit_loss:.2f} | Decision: {decision}"
            )

            if decision == "close":
                self.close_trade(pos["trade_id"], units)
            else:
                self.logger.info(
                    f"[analyze_open_trades] Holding {side.upper()} ID {pos.get('id', 'N/A')} at {current_price:.5f}."
                )

    def monitor_performance(self, accuracy):
        """Check rolling average accuracy; retrain if below threshold."""
        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) > 10:
            self.accuracy_history.pop(0)
        avg = sum(self.accuracy_history) / len(self.accuracy_history)
        self.logger.info(f"[monitor_performance] Rolling avg accuracy: {avg:.2f}")
        if avg < self.performance_threshold:
            self.logger.info("[monitor_performance] Below threshold, retraining.")
            self.retrain_model()

    def retrain_model(self):
        """Retrain ML model on schedule."""
        self.logger.info("[retrain_model] Starting retraining.")
        price_data = self.get_prices(count=500, timeframe="H1")
        if not price_data:
            self.logger.error("[retrain_model] No price data available.")
            return
        X_train, y_train, X_val, y_val = self.prepare_training_data(price_data)
        if X_train.size > 0 and y_train.size > 0 and X_val.size > 0 and y_val.size > 0:
            self.ml_model.train(X_train, y_train, X_val, y_val)
            self.logger.info("[retrain_model] Retraining complete.")
        else:
            self.logger.error("[retrain_model] Insufficient data for retraining.")

    def get_prices(self, count=500, timeframe="H1"):
        """Fetch candle data from OANDA."""
        for _ in range(self.retry_count):
            try:
                url = f"{BASE_URL}/instruments/{TRADE_INSTRUMENT}/candles"
                params = {"granularity": timeframe, "count": count}
                self.logger.info(f"[get_prices] Fetching candles from {url} with params {params}")
                r = requests.get(url, headers=self.headers, params=params, timeout=5)
                r.raise_for_status()
                data = r.json()
                if "candles" not in data or not data["candles"]:
                    self.logger.error("[get_prices] No candles data returned from OANDA.")
                    return None
                return data
            except requests.exceptions.HTTPError as http_err:
                self.logger.error(f"[get_prices] HTTP error occurred: {http_err}")
                return None  # 404 is a client error, likely incorrect endpoint
            except requests.exceptions.RequestException as e:
                self.logger.error(f"[get_prices] RequestException: {e}")
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"[get_prices] Unexpected error: {e}")
                time.sleep(1)
        self.logger.error("[get_prices] Failed after retries.")
        return None

    def make_trade(self, side, units, current_price, stop_loss, take_profit):
        # Defensive checks
        if stop_loss is None or take_profit is None:
            self.logger.error("[make_trade] Stop loss or take profit not set. Cannot proceed with trade.")
            return
        
        if not isinstance(stop_loss, (float, int)) or not isinstance(take_profit, (float, int)):
            self.logger.error("[make_trade] Stop loss and Take profit must be numeric values.")
            return
        
        for attempt in range(1, self.retry_count + 1):
            try:
                url = f"{BASE_URL}/accounts/{OANDA_ACCOUNT_ID}/orders"
                order_data = {
                    "order": {
                        "units": str(units) if side.lower() == "buy" else str(-units),
                        "instrument": TRADE_INSTRUMENT,
                        "timeInForce": "FOK",
                        "type": "MARKET",
                        "positionFill": "DEFAULT",
                        "stopLossOnFill": {
                            "price": f"{stop_loss:.5f}"
                        },
                        "takeProfitOnFill": {
                            "price": f"{take_profit:.5f}"
                        }
                    }
                }
                self.logger.info(f"[make_trade] Attempt {attempt}: Sending trade order to {url} with data {json.dumps(order_data)}")
                response = requests.post(url, headers=self.headers, json=order_data, timeout=5)
                self.logger.debug(f"[make_trade] Response Status Code: {response.status_code}")
                self.logger.debug(f"[make_trade] Response Text: {response.text}")
                if response.status_code in [200, 201]:
                    resp_json = response.json()
                    trade_id = resp_json.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
                    if not trade_id:
                        error_message = resp_json.get("errorMessage", "Trade ID not found and no error message provided.")
                        self.logger.error(f"[make_trade] Trade ID not found. Error: {error_message}. Full Response: {json.dumps(resp_json)}")
                        return
                    self.log_trade(resp_json)
                    self.load_account_balance()
                    
                    # Append the new position
                    self.positions.append({
                        "side": side.lower(),
                        "units": units,
                        "entry_price": current_price,
                        "trade_id": trade_id,
                        "sl_order_id": None,
                        "tp_order_id": None
                    })
                    self.open_position = True
                    self.current_side = side.lower()
                    self.entry_price = current_price
                    
                    self.logger.info(
                        f"[make_trade] {side.upper()} @ {current_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, Trade ID={trade_id}"
                    )
                    return
                else:
                    self.logger.error(f"[make_trade] OANDA error: {response.text}")
                    time.sleep(1)  # Wait before retrying
            except Exception as e:
                self.logger.error(f"[make_trade] Exception on attempt {attempt}: {e}")
                time.sleep(1)  # Wait before retrying
        self.logger.error("[make_trade] Failed after retries.")

    def close_position_fifo(self, units_to_close):
        remaining = units_to_close
        while remaining > 0 and self.positions:
            oldest = self.positions[0]
            if oldest["units"] <= remaining:
                pos = self.positions.pop(0)
                self._close_single_position(pos)
                remaining -= pos["units"]
            else:
                oldest["units"] -= remaining
                pos_copy = {
                    "side": oldest["side"],
                    "units": remaining,
                    "entry_price": oldest["entry_price"],
                    "trade_id": oldest["trade_id"],
                    "sl_order_id": oldest.get("sl_order_id"),
                    "tp_order_id": oldest.get("tp_order_id"),
                    "stop_loss": oldest["stop_loss"],
                    "take_profit": oldest["take_profit"]
                }
                self._close_single_position(pos_copy)
                remaining = 0

        if not self.positions:
            self.open_position = False
            self.current_side = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None

    def _close_single_position(self, pos):
        cp = self.get_current_price()
        if pos["side"] == "buy":
            realized = (cp - pos["entry_price"]) * pos["units"]
        else:
            realized = (pos["entry_price"] - cp) * pos["units"]
        self.profit_loss += realized
        self.logger.info(
            f"[_close_single_position] Closed {pos['units']} {pos['side'].upper()} @ {cp:.5f}. "
            f"Realized: {realized:.2f}, Total P/L: {self.profit_loss:.2f}"
        )
        self.save_historical_metrics()

    def cancel_order(self, order_id):
        """
        Cancel an existing order on OANDA.
        """
        try:
            url = f"{BASE_URL}/accounts/{self.account_id}/orders/{order_id}"
            self.logger.info(f"[cancel_order] Cancelling order {order_id} at {url}")
            response = requests.delete(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            self.logger.info(f"[cancel_order] Cancelled order ID: {order_id}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[cancel_order] API request failed: {e}")
        except Exception as e:
            self.logger.error(f"[cancel_order] Unexpected error: {e}")

    def can_trade(self, signal_type=None):
        """Enforce a cooldown between trades. Optionally pass a signal type for advanced logic."""
        if self.last_trade_time is None:
            return True
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        if elapsed < self.trade_cooldown:
            self.logger.info("[can_trade] Cooldown active, skipping trade.")
            return False
        return True

    def execute_trade(self, trade_signal):
        """Optional method to handle external signals."""
        if not self.can_trade():
            self.logger.info("[execute_trade] Skipped due to cooldown.")
            return
        self.last_trade_time = datetime.now()
        self.logger.info(f"[execute_trade] Trade signal executed: {trade_signal}")

    def get_current_price(self):
        price_data = self.get_prices(count=1)
        if price_data and "candles" in price_data:
            try:
                return float(price_data["candles"][-1]["mid"]["c"])
            except Exception as e:
                self.logger.error(f"[get_current_price] Extraction error: {e}")
                return 0.0
        self.logger.error("[get_current_price] Price data missing or invalid.")
        return 0.0

    def evaluate_indicators(self, price_data):
        """
        Calculate technical indicators and derive strategy-based signals.
        Return a dictionary with indicators and a summary of buy/sell signals.
        """
        closes = np.array([float(c['mid']['c']) for c in price_data['candles']])
        highs = np.array([float(c['mid']['h']) for c in price_data['candles']])
        lows = np.array([float(c['mid']['l']) for c in price_data['candles']])
        volumes = np.array([float(c['volume']) for c in price_data['candles']])

        # Calculate Indicators
        ema_20 = talib.EMA(closes, timeperiod=20)
        ema_50 = talib.EMA(closes, timeperiod=50)
        rsi = talib.RSI(closes, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        stochastic_k, stochastic_d = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
        boll_upper, boll_mid, boll_lower = talib.BBANDS(closes, timeperiod=20)
        atr = talib.ATR(highs, lows, closes, timeperiod=14)
        sar = talib.SAR(highs, lows, acceleration=0.02, maximum=0.2)
        williams_r = talib.WILLR(highs, lows, closes, timeperiod=14)

        # Strategies
        strategy_signals = {
            "trend_following": {
                "buy": (ema_20[-1] > ema_50[-1]) and (closes[-1] > sar[-1]),
                "sell": (ema_20[-1] < ema_50[-1]) and (closes[-1] < sar[-1]),
            },
            "mean_reversion": {
                "buy": (rsi[-1] < 30) and (closes[-1] < boll_lower[-1]),
                "sell": (rsi[-1] > 70) and (closes[-1] > boll_upper[-1]),
            },
            "breakout": {
                "buy": (closes[-1] > highs[-2]) and (volumes[-1] > np.mean(volumes[-10:])),
                "sell": (closes[-1] < lows[-2]) and (volumes[-1] > np.mean(volumes[-10:])),
            },
            "momentum": {
                "buy": (macd[-1] > macd_signal[-1]) and (stochastic_k[-1] > stochastic_d[-1]) and (williams_r[-1] < -80),
                "sell": (macd[-1] < macd_signal[-1]) and (stochastic_k[-1] < stochastic_d[-1]) and (williams_r[-1] > -20),
            },
            "range_bound": {
                "buy": (closes[-1] < boll_lower[-1]) and (40 <= rsi[-1] <= 60),
                "sell": (closes[-1] > boll_upper[-1]) and (40 <= rsi[-1] <= 60),
            },
        }

        # Summaries
        buy_signals = sum(s["buy"] for s in strategy_signals.values())
        sell_signals = sum(s["sell"] for s in strategy_signals.values())

        results = {
            "indicators": {
                "ema_20": ema_20[-1],
                "ema_50": ema_50[-1],
                "rsi": rsi[-1],
                "macd": macd[-1],
                "macd_signal": macd_signal[-1],
                "macd_hist": macd_hist[-1],
                "stochastic_k": stochastic_k[-1],
                "stochastic_d": stochastic_d[-1],
                "boll_upper": boll_upper[-1],
                "boll_mid": boll_mid[-1],
                "boll_lower": boll_lower[-1],
                "atr": atr[-1],
                "sar": sar[-1],
                "williams_r": williams_r[-1],
            },
            "strategy_signals": strategy_signals,
            "summary": {
                "buy_signals": buy_signals,
                "sell_signals": sell_signals
            }
        }

        self.logger.info(f"[evaluate_indicators] Summary: {results['summary']}")
        return results

    def manage_position(self, current_price, prediction, indicators):
        """
        Manage open positions or place new trades based on indicator summaries and ML prediction.
        """
        self.synchronize_positions()
        if not self.positions:
            self.logger.info("[analyze_open_trades] No open positions to analyze.")
        
        # Ensure we do not duplicate trade executions
        buy_signals = indicators["summary"]["buy_signals"]
        sell_signals = indicators["summary"]["sell_signals"]

        # ML adds extra weight
        if prediction == 1:
            buy_signals += 2
        elif prediction == 0:
            sell_signals += 2

        confidence_threshold = 3
        if not self.open_position:
            if buy_signals >= confidence_threshold:
                self.logger.info(f"[manage_position] Opening BUY. Confidence score: {buy_signals}")
                # Calculate Stop Loss and Take Profit
                stop_loss = current_price - (current_price * (self.stop_loss_percentage / 100))
                take_profit = current_price + (current_price * (self.take_profit_percentage / 100))
                self.logger.info(f"[manage_position] Calculated SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
                self.make_trade("buy", self.position_units, current_price, stop_loss, take_profit)
            elif sell_signals >= confidence_threshold:
                self.logger.info(f"[manage_position] Opening SELL. Confidence score: {sell_signals}")
                # Calculate Stop Loss and Take Profit
                stop_loss = current_price + (current_price * (self.stop_loss_percentage / 100))
                take_profit = current_price - (current_price * (self.take_profit_percentage / 100))
                self.logger.info(f"[manage_position] Calculated SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
                self.make_trade("sell", self.position_units, current_price, stop_loss, take_profit)
            else:
                self.logger.info("[manage_position] No strong signals to open a new position.")

        if indicators["indicators"]["rsi"] > 80 or indicators["indicators"]["rsi"] < 20:
            self.logger.info("[manage_position] RSI in extreme zone. Skipping trade to avoid overbought/oversold conditions.")
            return

        if self.open_position:
            # If there's a flip or SL/TP triggered
            if ((self.current_side == "buy" and sell_signals >= confidence_threshold) or
                (self.current_side == "sell" and buy_signals >= confidence_threshold)):
                self.logger.info("[manage_position] Signal flip. Closing current position.")
                self.close_position_fifo(self.position_units)
            elif self.stop_loss and self.take_profit:
                if self.current_side == "buy":
                    if current_price <= self.stop_loss or current_price >= self.take_profit:
                        self.logger.info("[manage_position] SL/TP triggered for BUY => closing.")
                        self.close_position_fifo(self.position_units)
                else:  # sell
                    if current_price >= self.stop_loss or current_price <= self.take_profit:
                        self.logger.info("[manage_position] SL/TP triggered for SELL => closing.")
                        self.close_position_fifo(self.position_units)
            else:
                self.logger.info("[manage_position] Holding position.")
                self.adjust_trailing_stop(current_price)

    def apply_settings(self, settings):
        """
        Apply settings from the BotManager to the TradingBot.
        """
        try:
            self.risk_level = settings.get("riskLevel", self.risk_level)
            self.max_trades = int(settings.get("maxTrades", self.max_trades))
            self.trade_interval_hours = float(settings.get("tradeIntervalHours", self.trade_interval_hours))
            self.stop_loss_percentage = float(settings.get("stopLossPercentage", self.stop_loss_percentage))
            self.take_profit_percentage = float(settings.get("takeProfitPercentage", self.take_profit_percentage))
            self.trailing_atr_multiplier = float(settings.get("trailingATRMultiplier", self.trailing_atr_multiplier))
            self.adjust_atr_multiplier = float(settings.get("adjustATRMultiplier", self.adjust_atr_multiplier))
            self.trade_cooldown = int(settings.get("tradeCooldown", self.trade_cooldown))
            self.logger.info("TradingBot: Settings applied successfully.")
        except Exception as e:
            self.logger.error(f"TradingBot: Failed to apply settings - {e}")

    def adjust_trailing_stop(self, current_price):
        """Dynamically move SL based on ATR to secure profits."""
        self.logger.info("[adjust_trailing_stop] Starting trailing stop adjustment based on ATR.")
        try:
            # Fetch recent price data
            price_data = self.get_prices(count=100, timeframe="H1")  # Adjust count and timeframe as needed
            if not price_data:
                self.logger.error("[adjust_trailing_stop] Failed to fetch price data.")
                return

            # Extract price arrays
            closes = np.array([float(c['mid']['c']) for c in price_data['candles']])
            highs = np.array([float(c['mid']['h']) for c in price_data['candles']])
            lows = np.array([float(c['mid']['l']) for c in price_data['candles']])
            volumes = np.array([float(c['volume']) for c in price_data['candles']])

            # Calculate ATR
            atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]

            # Current position side
            if not self.positions:
                self.logger.warning("[adjust_trailing_stop] No open positions to adjust.")
                return

            # Iterate through all open positions
            for pos in self.positions:
                side = pos['side']
                trade_id = pos['trade_id']
                sl_order_id = pos.get("sl_order_id")

                if not trade_id:
                    self.logger.error("[adjust_trailing_stop] Trade ID not found for a position.")
                    continue

                multiplier = self.trailing_atr_multiplier  # User-configurable multiplier

                if side == "buy":
                    # Calculate new SL
                    new_sl = current_price - (atr * multiplier)
                    if new_sl > pos['stop_loss']:
                        self.logger.info(f"[adjust_trailing_stop] Tightening SL from {pos['stop_loss']:.5f} to {new_sl:.5f} for trade {trade_id}.")
                        # Cancel existing SL order
                        if sl_order_id:
                            self.cancel_order(sl_order_id)
                        # Set new SL
                        self.set_stop_loss_take_profit(trade_id, new_sl, pos['take_profit'])
                        pos['stop_loss'] = new_sl
                    else:
                        self.logger.info(f"[adjust_trailing_stop] SL not adjusted for trade {trade_id}. Current SL: {pos['stop_loss']:.5f}, New SL: {new_sl:.5f}")

                elif side == "sell":
                    # Calculate new SL
                    new_sl = current_price + (atr * multiplier)
                    if new_sl < pos['stop_loss']:
                        self.logger.info(f"[adjust_trailing_stop] Tightening SL from {pos['stop_loss']:.5f} to {new_sl:.5f} for trade {trade_id}.")
                        # Cancel existing SL order
                        if sl_order_id:
                            self.cancel_order(sl_order_id)
                        # Set new SL
                        self.set_stop_loss_take_profit(trade_id, new_sl, pos['take_profit'])
                        pos['stop_loss'] = new_sl
                    else:
                        self.logger.info(f"[adjust_trailing_stop] SL not adjusted for trade {trade_id}. Current SL: {pos['stop_loss']:.5f}, New SL: {new_sl:.5f}")

                else:
                    self.logger.warning(f"[adjust_trailing_stop] Unknown position side: {side}")

                self.logger.info(f"[adjust_trailing_stop] Updated SL: {pos['stop_loss']:.5f}, TP: {pos['take_profit']:.5f}")

        except Exception as e:
            self.logger.error(f"[adjust_trailing_stop] Exception: {e}")

    def set_stop_loss_take_profit(self, trade_id, new_sl, new_tp):
        """Set new Stop Loss and Take Profit orders for a specific trade and store their order IDs."""
        try:
            # Set Stop Loss
            sl_url = f"{BASE_URL}/accounts/{self.account_id}/orders"
            sl_order = {
                "order": {
                    "type": "STOP_LOSS",
                    "tradeID": trade_id,
                    "price": f"{new_sl:.5f}",
                    "timeInForce": "GTC",
                    "triggerCondition": "DEFAULT"
                }
            }
            self.logger.info(f"[set_stop_loss_take_profit] Setting Stop Loss for trade {trade_id} at {sl_url} with data {sl_order}")
            sl_response = requests.post(sl_url, headers=self.headers, json=sl_order, timeout=5)
            if sl_response.status_code in [200, 201]:
                sl_order_id = sl_response.json().get("orderFillTransaction", {}).get("id")
                self.logger.info(f"[set_stop_loss_take_profit] Stop Loss set to {new_sl:.5f} for trade {trade_id}. Order ID: {sl_order_id}")
                # Update the position with the new SL order ID
                for pos in self.positions:
                    if pos['trade_id'] == trade_id:
                        pos['sl_order_id'] = sl_order_id
                        break
            else:
                self.logger.error(f"[set_stop_loss_take_profit] Failed to set Stop Loss: {sl_response.text}")

            # Set Take Profit
            tp_url = f"{BASE_URL}/accounts/{self.account_id}/orders"
            tp_order = {
                "order": {
                    "type": "TAKE_PROFIT",
                    "tradeID": trade_id,
                    "price": f"{new_tp:.5f}",
                    "timeInForce": "GTC",
                    "triggerCondition": "DEFAULT"
                }
            }
            self.logger.info(f"[set_stop_loss_take_profit] Setting Take Profit for trade {trade_id} at {tp_url} with data {tp_order}")
            tp_response = requests.post(tp_url, headers=self.headers, json=tp_order, timeout=5)
            if tp_response.status_code in [200, 201]:
                tp_order_id = tp_response.json().get("orderFillTransaction", {}).get("id")
                self.logger.info(f"[set_stop_loss_take_profit] Take Profit set to {new_tp:.5f} for trade {trade_id}. Order ID: {tp_order_id}")
                # Update the position with the new TP order ID
                for pos in self.positions:
                    if pos['trade_id'] == trade_id:
                        pos['tp_order_id'] = tp_order_id
                        break
            else:
                self.logger.error(f"[set_stop_loss_take_profit] Failed to set Take Profit: {tp_response.text}")

        except Exception as e:
            self.logger.error(f"[set_stop_loss_take_profit] Exception: {e}")


    def adjust_take_profit_stop_loss(self, current_price):
        """
        Tighten SL/TP based on Bollinger Bands and ATR to secure profits and limit losses.
        This method updates both the internal variables and OANDA's orders.
        """
        self.logger.info("[adjust_take_profit_stop_loss] Starting SL/TP adjustment based on Bollinger Bands and ATR.")
        try:
            # Fetch recent price data
            price_data = self.get_prices(count=100, timeframe="H1")  # Adjust count and timeframe as needed
            if not price_data:
                self.logger.error("[adjust_take_profit_stop_loss] Failed to fetch price data.")
                return

            # Extract price arrays
            closes = np.array([float(c['mid']['c']) for c in price_data['candles']])
            highs = np.array([float(c['mid']['h']) for c in price_data['candles']])
            lows = np.array([float(c['mid']['l']) for c in price_data['candles']])
            volumes = np.array([float(c['volume']) for c in price_data['candles']])

            # Calculate indicators
            atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
            boll_upper, boll_mid, boll_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)

            # Current position side
            if not self.positions:
                self.logger.warning("[adjust_take_profit_stop_loss] No open positions to adjust.")
                return

            # Iterate through all open positions
            for pos in self.positions:
                side = pos['side']
                trade_id = pos['trade_id']
                sl_order_id = pos.get("sl_order_id")
                tp_order_id = pos.get("tp_order_id")

                if not trade_id:
                    self.logger.error("[adjust_take_profit_stop_loss] Trade ID not found for a position.")
                    continue

                if side == "buy":
                    # Calculate new SL
                    new_sl = current_price - (atr * self.adjust_atr_multiplier)
                    if new_sl > pos['stop_loss']:
                        self.logger.info(f"[adjust_take_profit_stop_loss] Tightening SL from {pos['stop_loss']:.5f} to {new_sl:.5f} for trade {trade_id}.")
                        # Cancel existing SL order
                        if sl_order_id:
                            self.cancel_order(sl_order_id)
                        # Set new SL
                        self.set_stop_loss_take_profit(trade_id, new_sl, pos['take_profit'])
                        pos['stop_loss'] = new_sl
                    else:
                        self.logger.info(f"[adjust_take_profit_stop_loss] SL not adjusted for trade {trade_id}. Current SL: {pos['stop_loss']:.5f}, New SL: {new_sl:.5f}")

                    # Calculate new TP
                    new_tp = min(current_price + (atr * (self.adjust_atr_multiplier * 2)), boll_upper[-1])
                    if new_tp > pos['take_profit']:
                        self.logger.info(f"[adjust_take_profit_stop_loss] Adjusting TP from {pos['take_profit']:.5f} to {new_tp:.5f} for trade {trade_id}.")
                        # Cancel existing TP order
                        if tp_order_id:
                            self.cancel_order(tp_order_id)
                        # Set new TP
                        self.set_stop_loss_take_profit(trade_id, pos['stop_loss'], new_tp)
                        pos['take_profit'] = new_tp
                    else:
                        self.logger.info(f"[adjust_take_profit_stop_loss] TP not adjusted for trade {trade_id}. Current TP: {pos['take_profit']:.5f}, New TP: {new_tp:.5f}")

                elif side == "sell":
                    # Calculate new SL
                    new_sl = current_price + (atr * self.adjust_atr_multiplier)
                    if new_sl < pos['stop_loss']:
                        self.logger.info(f"[adjust_take_profit_stop_loss] Tightening SL from {pos['stop_loss']:.5f} to {new_sl:.5f} for trade {trade_id}.")
                        # Cancel existing SL order
                        if sl_order_id:
                            self.cancel_order(sl_order_id)
                        # Set new SL
                        self.set_stop_loss_take_profit(trade_id, new_sl, pos['take_profit'])
                        pos['stop_loss'] = new_sl
                    else:
                        self.logger.info(f"[adjust_take_profit_stop_loss] SL not adjusted for trade {trade_id}. Current SL: {pos['stop_loss']:.5f}, New SL: {new_sl:.5f}")

                    # Calculate new TP
                    new_tp = max(current_price - (atr * (self.adjust_atr_multiplier * 2)), boll_lower[-1])
                    if new_tp < pos['take_profit']:
                        self.logger.info(f"[adjust_take_profit_stop_loss] Adjusting TP from {pos['take_profit']:.5f} to {new_tp:.5f} for trade {trade_id}.")
                        # Cancel existing TP order
                        if tp_order_id:
                            self.cancel_order(tp_order_id)
                        # Set new TP
                        self.set_stop_loss_take_profit(trade_id, pos['stop_loss'], new_tp)
                        pos['take_profit'] = new_tp
                    else:
                        self.logger.info(f"[adjust_take_profit_stop_loss] TP not adjusted for trade {trade_id}. Current TP: {pos['take_profit']:.5f}, New TP: {new_tp:.5f}")

                else:
                    self.logger.warning(f"[adjust_take_profit_stop_loss] Unknown position side: {side}")

            self.logger.info("[adjust_take_profit_stop_loss] Completed SL/TP adjustments.")

        except Exception as e:
            self.logger.error(f"[adjust_trailing_stop] Exception: {e}")

    def determine_signal(self, price_data):
        """
        Quick check using ML predictions to confirm final buy/sell signals.
        """
        feats = self.ml_model.prepare_features(price_data)
        if feats.shape[0] == 0:
            return None

        ml_pred = self.ml_model.predict(feats[-1].reshape(1, -1))
        if ml_pred is None:
            return None  # Invalid

        # Additional quick check (like RSI)
        closes = [float(c['mid']['c']) for c in price_data['candles']]
        rsi = talib.RSI(np.array(closes), timeperiod=14)
        if rsi.size == 0:
            self.logger.info("[determine_signal] RSI not found or insufficient data.")
            return None

        final_signal = ml_pred[0]
        self.logger.info(f"[determine_signal] ML suggests {final_signal}. RSI: {rsi[-1]}")
        return final_signal

    def run_job_queue(self):
        """
        Periodically run logic to ensure no missed trades or analysis steps.
        """
        try:
            price_data = self.get_prices(count=500, timeframe="H1")  # Specify timeframe
            if not price_data:
                self.logger.info("[run_job_queue] No price data available.")
                return

            try:
                indicators = self.evaluate_indicators(price_data)
            except Exception as e:
                self.logger.error(f"[run_job_queue] Indicator eval error: {e}")
                return

            # Prepare ML features
            features = self.ml_model.prepare_features(price_data)
            if features.shape[0] == 0:
                self.logger.info("[run_job_queue] No valid features for ML.")
                return

            prediction = self.ml_model.predict(features[-1].reshape(1, -1))  # Ensure correct input shape
            if prediction is None:
                self.logger.info("[run_job_queue] ML returned invalid prediction.")
                return

            current_price = self.get_current_price()
            if current_price == 0:
                self.logger.info("[run_job_queue] Unable to fetch current price.")
                return
            
            # Multi-timeframe signals
            buy_signals, sell_signals = self.multi_timeframe_analysis()

            # Combine them into the dictionary-based indicators for manage_position
            # so we can pass the final buy/sell signals
            combined_indicators = {
                "indicators": indicators["indicators"],
                "summary": {
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals
                }
            }

            self.analyze_open_trades()

            # Assuming calculate_indicators is separate from evaluate_indicators
            indicators_calculated = self.calculate_indicators()

            if not self.open_position and self.can_trade():
                self.manage_position(current_price, prediction, combined_indicators)

            # Adjust SL/TP after managing positions
            if self.open_position:
                self.adjust_trailing_stop(current_price)
                # Optionally, adjust SL/TP based on Bollinger Bands
                self.adjust_take_profit_stop_loss(current_price)

        except Exception as e:
            self.logger.error(f"[run_job_queue] Unexpected error: {e}")

    def run(self):
        """Alternative continuous loop (if not using external scheduling)."""
        self.running = True
        self.start_time = datetime.now()
        self.logger.info("[run] Trading bot started in continuous loop.")
        try:
            while self.running:
                data = self.get_prices()
                if not data:
                    time.sleep(5)
                    continue

                feats = self.ml_model.prepare_features(data)
                if feats.size == 0:
                    time.sleep(10)
                    continue

                prediction = self.ml_model.predict(feats[-1].reshape(1, -1))
                if prediction in [0, 1] and self.can_trade():
                    cp = self.get_current_price()
                    # Minimal indicators:
                    combined_indicators = {"summary": {"buy_signals": 1 if prediction[0] == 1 else 0,
                                                      "sell_signals": 1 if prediction[0] == 0 else 0}}
                    self.manage_position(cp, prediction, combined_indicators)

                    # Adjust SL/TP after managing positions
                    if self.open_position:
                        self.adjust_trailing_stop(cp)
                        # Optionally, adjust SL/TP based on Bollinger Bands
                        self.adjust_take_profit_stop_loss(cp)

                time.sleep(10)
        except Exception as e:
            self.logger.error(f"[run] Error in continuous loop: {e}")
        finally:
            self.logger.info("[run] Trading bot stopped.")

    def prepare_training_data(self, price_data):
        """Convert OANDA data -> (X, y) for ML training."""
        features = self.ml_model.prepare_features(price_data, training=True)
        closes = [float(c['mid']['c']) for c in price_data['candles']]
        closes = closes[-len(features):]
        labels = [(1 if closes[i] > closes[i - 1] else 0) for i in range(1, len(closes))]
        
        X = features[1:]
        y = np.array(labels)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        return X_train, y_train, X_val, y_val

    def load_state(self):
        """
        Load the bot's state from the SQLite database.
        """
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY,
                    running BOOLEAN,
                    start_time TEXT
                )
            ''')
            cursor.execute('SELECT running, start_time FROM bot_state WHERE id=1')
            row = cursor.fetchone()
            if row:
                self.running, start_time_str = row
                self.start_time = datetime.fromisoformat(start_time_str) if start_time_str else None
                self.logger.info("[load_state] Loaded existing state from database.")
            else:
                cursor.execute('INSERT INTO bot_state (id, running, start_time) VALUES (1, ?, ?)', (self.running, self.start_time.isoformat() if self.start_time else None))
                conn.commit()
                self.logger.info("[load_state] Initialized new state in database.")
            conn.close()
        except Exception as e:
            self.logger.error(f"[load_state] Failed to load state: {e}")

    def save_state(self):
        """
        Save the bot's current state to the SQLite database.
        """
        try:
            conn = sqlite3.connect('metrics.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY,
                    running BOOLEAN,
                    start_time TEXT
                )
            ''')
            cursor.execute('''
                UPDATE bot_state SET running=?, start_time=? WHERE id=1
            ''', (self.running, self.start_time.isoformat() if self.start_time else None))
            conn.commit()
            conn.close()
            self.logger.info("[save_state] State saved to database.")
        except Exception as e:
            self.logger.error(f"[save_state] Failed to save state: {e}")

    def start(self):
        """
        Mark the bot as running and save the state.
        """
        self.running = True
        self.start_time = datetime.now()
        self.logger.info("[start] TradingBot has started running.")
        self.save_state()

    def stop(self):
        """
        Stop the bot, save the state, and shut down the scheduler.
        """
        self.running = False
        self.start_time = None
        self.save_state()
        try:
            self.scheduler.shutdown()
        except Exception as e:
            self.logger.error(f"[stop] Error shutting down scheduler: {e}")
        self.logger.info("[stop] TradingBot stopped.")

    def log_trade(self, resp_data):
        """Logs a new trade from OANDA's orderFillTransaction."""
        if 'orderFillTransaction' in resp_data:
            f = resp_data['orderFillTransaction']
            trade_id = f.get("tradeOpened", {}).get("tradeID")
            if not trade_id:
                self.logger.error("[log_trade] Trade ID not found in orderFillTransaction.")
                return
            tr = {
                'id': trade_id,
                'time': f['time'],
                'instrument': f['instrument'],
                'units': f['units'],
                'price': float(f['price']),
                'sl_order_id': None,  # To be updated when SL is set
                'tp_order_id': None   # To be updated when TP is set
            }
            self.trades.append(tr)
            self.trade_count += 1
            self.logger.info(f"[log_trade] Recorded new trade: {tr}")
            self.save_historical_metrics()

    def get_trades(self):
        """Return local list of trades."""
        return self.trades

    def get_open_trades(self):
        """
        Fetch open trades from OANDA with enhanced error handling.
        """
        try:
            url = f"{BASE_URL}/accounts/{self.account_id}/openTrades"
            self.logger.info(f"[get_open_trades] Fetching open trades from {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            trades = response.json().get("trades", [])
            for trade in trades:
                trade["price"] = float(trade.get("price", 0))
                trade["currentUnits"] = float(trade.get("currentUnits", 0))
                trade["tradeID"] = trade.get("id")  # Ensure 'tradeID' is available
            self.logger.info(f"[get_open_trades] {len(trades)} open trades found on OANDA.")
            return trades
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[get_open_trades] API request failed: {e}")
            return []
        except ValueError as e:
            self.logger.error(f"[get_open_trades] JSON decoding failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"[get_open_trades] Unexpected error: {e}")
            return []

    def process_open_trades(self):
        """Example method for analyzing or closing OANDA open trades."""
        open_trades = self.get_open_trades()
        if not open_trades:
            self.logger.info("[process_open_trades] No open trades to process.")
            return
        current_price = self.get_current_price()
        for trade in open_trades:
            trade_id = trade.get("id")
            side = "buy" if float(trade.get("currentUnits", 0)) > 0 else "sell"
            entry_price = float(trade.get("price", 0))
            units = abs(int(float(trade.get("currentUnits", 0))))
            if self.should_hold_trade(current_price, side, entry_price):
                self.logger.info(f"[process_open_trades] Holding {side.upper()} {trade_id}.")
            else:
                self.logger.info(f"[process_open_trades] Closing {side.upper()} {trade_id} for {units} units.")
                self.close_trade(trade_id, units)

    def should_hold_trade(self, current_price, side, entry_price):
        """Simple hold example if profit margin > 1%."""
        if side == "buy":
            profit_margin = (current_price - entry_price) / entry_price
        else:
            profit_margin = (entry_price - current_price) / entry_price
        return profit_margin > 0.01

    def close_trade(self, trade_id, units):
        """Close a trade on OANDA by ID, partial or full."""
        try:
            url = f"{BASE_URL}/accounts/{self.account_id}/trades/{trade_id}/close"
            data = {"units": str(units)}
            self.logger.info(f"[close_trade] Closing trade {trade_id} with data {data} at {url}")
            response = requests.put(url, headers=self.headers, json=data)
            response.raise_for_status()
            self.logger.info(f"[close_trade] Closed trade {trade_id} for {units} units on OANDA.")
            self.load_account_balance()
        except Exception as e:
            self.logger.error(f"[close_trade] Error: {e}")

    def get_oanda_trade_history(self):
        """Fetch historical trades from OANDA."""
        try:
            url = f"{BASE_URL}/accounts/{self.account_id}/trades"
            self.logger.info(f"[get_oanda_trade_history] Fetching trade history from {url}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            trades = response.json().get("trades", [])
            for trade in trades:
                trade["price"] = float(trade.get("price", 0))
                trade["currentUnits"] = float(trade.get("currentUnits", 0))
            self.logger.info(f"[get_oanda_trade_history] {len(trades)} trades found in OANDA history.")
            return trades
        except Exception as e:
            self.logger.error(f"[get_oanda_trade_history] Error: {e}")
            return []

    def get_metrics(self):
        elapsed_time = str(datetime.now() - self.start_time) if self.start_time else "00:00:00"
        return {
            "accountBalance": self.account_balance,
            "tradeCount": len(self.trades),
            "profitLoss": self.profit_loss,
            "timeElapsed": elapsed_time
        }
    
    def calculate_indicators(self):
        """
        Calculate technical indicators.
        """
        # Implement indicator calculations using TA-Lib or similar
        # Example:
        price_data = self.get_prices(count=100, timeframe="H1")
        closes = [float(candle['mid']['c']) for candle in price_data['candles']]
        rsi = talib.RSI(np.array(closes), timeperiod=14)
        ema_fast = talib.EMA(np.array(closes), timeperiod=12)
        ema_slow = talib.EMA(np.array(closes), timeperiod=26)
        bbands_upper, bbands_middle, bbands_lower = talib.BBANDS(np.array(closes), timeperiod=20)
        
        # Determine buy/sell signals based on indicators
        buy_signals = 0
        sell_signals = 0
        if ema_fast[-1] > ema_slow[-1]:
            buy_signals += 1
        if ema_fast[-1] < ema_slow[-1]:
            sell_signals += 1
        if rsi[-1] < 30:
            buy_signals += 1
        if rsi[-1] > 70:
            sell_signals += 1
        # Add more indicator-based signals as needed
        
        return {
            "summary": {
                "buy_signals": buy_signals,
                "sell_signals": sell_signals
            },
            "indicators": {
                "rsi": rsi[-1],
                "ema_fast": ema_fast[-1],
                "ema_slow": ema_slow[-1],
                "bbands_upper": bbands_upper[-1],
                "bbands_middle": bbands_middle[-1],
                "bbands_lower": bbands_lower[-1]
            }
        }

    def calculate_profit_loss(self):
        return sum(trade.get("profit", 0) for trade in self.trades)
