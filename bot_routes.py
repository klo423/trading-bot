# bot/routes.py
from flask import Flask, jsonify, request, render_template
from bot_trader import Trader
from bot import config

app = Flask(__name__)
trader = Trader()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/status')
def status():
    positions = trader.api.list_positions()
    trades = trader.db.cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20").fetchall()
    
    return jsonify({
        'success': True,
        'positions': [{
            'symbol': p.symbol,
            'qty': p.qty,
            'avg_price': p.avg_entry_price,
            'current_price': p.current_price,
            'market_value': p.market_value,
            'unrealized_pl': p.unrealized_pl
        } for p in positions],
        'trades': [
            {
                'symbol': t[1],
                'side': t[2],
                'qty': t[3],
                'price': t[4],
                'timestamp': t[5]
            } for t in trades
        ]
    })

@app.route('/api/trade', methods=['POST'])
def trade():
    data = request.json
    symbol = data.get('symbol')
    side = data.get('side')
    qty = data.get('qty', config.POSITION_SIZE)
    order = trader.submit_trade(symbol, side, qty)
    return jsonify({'success': order is not None})
