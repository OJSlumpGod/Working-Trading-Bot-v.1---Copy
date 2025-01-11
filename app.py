from flask import Flask, render_template, jsonify, request, Response
from bot_manager import BotManager
import json
import sqlite3
import time
from datetime import datetime
import logging
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize BotManager
bot_manager = BotManager()

# Configuration
SETTINGS_FILE = 'settings.json'
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure Flask Logger
flask_logger = logging.getLogger('flask_app')
flask_logger.setLevel(logging.INFO)

# File handler for Flask logs
flask_file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'flask_app.log'))
flask_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
flask_file_handler.setFormatter(flask_formatter)
flask_logger.addHandler(flask_file_handler)

# Console handler for Flask logs
console_handler = logging.StreamHandler()
console_handler.setFormatter(flask_formatter)
flask_logger.addHandler(console_handler)

def load_settings():
    """Load settings from file."""
    try:
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        default_settings = {
            "riskLevel": "medium",
            "maxTrades": 5,
            "tradeIntervalHours": 24,
            "stopLossPercentage": 2.0,
            "takeProfitPercentage": 5.0,
            "trailingATRMultiplier": 1.5,
            "adjustATRMultiplier": 1.5,
            "tradeCooldown": 60
        }
        save_settings(default_settings)
        return default_settings
    except json.JSONDecodeError as e:
        flask_logger.error(f"JSON decode error in settings file: {e}")
        return {
            "riskLevel": "medium",
            "maxTrades": 5,
            "tradeIntervalHours": 24,
            "stopLossPercentage": 2.0,
            "takeProfitPercentage": 5.0,
            "trailingATRMultiplier": 1.5,
            "adjustATRMultiplier": 1.5,
            "tradeCooldown": 60
        }

def save_settings(data):
    """Save settings to file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        flask_logger.info("Settings saved successfully.")
    except Exception as e:
        flask_logger.error(f"Failed to save settings: {e}")

def initialize_bot():
    """Initialize bot with settings from file."""
    settings = load_settings()
    bot_manager.apply_settings(settings)
    flask_logger.info("Bot initialized with settings.")

# Initialize bot settings on startup
initialize_bot()

@app.route("/")
def index():
    """Render the main overview page."""
    return render_template("overview.html")

@app.route("/metrics_stream")  # Updated endpoint name to match front-end
def metrics_stream():
    """
    SSE endpoint to stream the bot’s metrics (trade count, P/L, time elapsed, etc.).
    """
    def event_stream():
        while True:
            try:
                metrics = bot_manager.get_metrics()
                yield f"data: {json.dumps(metrics)}\n\n"
                time.sleep(1)
            except GeneratorExit:
                flask_logger.info("Client disconnected from metrics_stream stream.")
                break
            except Exception as e:
                flask_logger.error(f"Error in metrics_stream stream: {e}")
                break
    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/history_data", methods=["GET"])
def history_data():
    """
    Return local trades (tracked by the trading bot) and OANDA trades (fetched from broker).
    """
    try:
        oanda_trades = bot_manager.trading_bot.get_oanda_trade_history()
        local_trades = bot_manager.trading_bot.get_trades()
        return jsonify({
            "oandaTrades": oanda_trades,
            "localTrades": local_trades
        }), 200
    except Exception as e:
        flask_logger.error(f"Error in history_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/open_positions", methods=["GET"])
def open_positions():
    """
    Return current open trades from OANDA.
    """
    try:
        open_positions = bot_manager.trading_bot.get_open_trades()
        return jsonify(open_positions), 200
    except Exception as e:
        flask_logger.error(f"Error in open_positions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/overview")
def overview():
    """Render the overview page."""
    return render_template("overview.html")

@app.route("/positions")
def positions():
    """Render the positions page."""
    return render_template("positions.html")

@app.route("/history")
def history():
    """Render the history page."""
    return render_template("history.html")

@app.route("/settings")
def settings_page():
    """Render the settings page."""
    return render_template("settings.html")

@app.route("/settings", methods=["GET"])
def get_settings():
    """Fetch current settings from the JSON file."""
    try:
        settings = load_settings()
        return jsonify(settings), 200
    except Exception as e:
        flask_logger.error(f"Error fetching settings: {e}")
        return jsonify({"error": f"Failed to fetch settings: {str(e)}"}), 500

@app.route("/settings", methods=["POST"])
def update_settings():
    """
    Update settings (risk level, max trades, intervals, SL/TP, etc.) in settings.json
    and pass them to the bot.
    """
    try:
        new_settings = request.json
        if not new_settings:
            raise ValueError("No settings data provided.")

        # Validate settings (optional but recommended)
        required_fields = ["riskLevel", "maxTrades", "tradeIntervalHours",
                           "stopLossPercentage", "takeProfitPercentage",
                           "trailingATRMultiplier", "adjustATRMultiplier",
                           "tradeCooldown"]
        for field in required_fields:
            if field not in new_settings:
                raise ValueError(f"Missing required setting: {field}")

        save_settings(new_settings)
        bot_manager.apply_settings(new_settings)
        return jsonify({"status": "Settings applied successfully"}), 200
    except ValueError as ve:
        flask_logger.warning(f"Validation error in update_settings: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        flask_logger.error(f"Error applying settings: {str(e)}")
        return jsonify({"error": f"Failed to apply settings: {str(e)}"}), 500

@app.route("/start_bot", methods=["POST"])
def start_bot():
    """
    Start the trading bot if not already running.
    """
    try:
        if not bot_manager.running:
            bot_manager.start_bot()
            return jsonify({"success": True, "message": "Bot started successfully."}), 200
        return jsonify({"success": False, "message": "Bot is already running."}), 200
    except Exception as e:
        flask_logger.error(f"Error starting bot: {e}")
        return jsonify({"success": False, "message": f"Failed to start bot: {str(e)}"}), 500

@app.route("/stop_bot", methods=["POST"])
def stop_bot():
    """
    Stop the trading bot if it's currently running.
    """
    try:
        if bot_manager.running:
            bot_manager.stop_bot()
            return jsonify({"success": True, "message": "Bot stopped successfully."}), 200
        return jsonify({"success": False, "message": "Bot is not running."}), 200
    except Exception as e:
        flask_logger.error(f"Error stopping bot: {e}")
        return jsonify({"success": False, "message": f"Failed to stop bot: {str(e)}"}), 500

@app.route("/reset_bot", methods=["POST"])
def reset_bot():
    """
    Reset the trading bot.
    """
    try:
        bot_manager.reset_bot()  # Ensure this method is correctly implemented in BotManager
        return jsonify({"status": "success", "message": "Bot reset successfully!"}), 200
    except Exception as e:
        flask_logger.error(f"Error resetting bot: {e}")
        return jsonify({"status": "error", "message": f"Failed to reset bot: {str(e)}"}), 500


if __name__ == "__main__":
    # Ensure metrics table exists
    try:
        conn = sqlite3.connect('metrics.db')
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_count INT,
                profit_loss FLOAT
            )
        """)
        conn.commit()
        conn.close()
        flask_logger.info("Metrics table ensured in DB.")
    except Exception as e:
        flask_logger.error(f"Error ensuring metrics table: {e}")

    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
