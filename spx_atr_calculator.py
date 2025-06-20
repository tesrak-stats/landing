import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import numpy as np

class SPYLevelsCalculator:
    def __init__(self, api_key, data_file='spy_levels_data.json'):
        self.api_key = api_key
        self.data_file = data_file
        self.et_tz = pytz.timezone('US/Eastern')
        
        # Fibonacci ratios for levels
        self.fib_ratios = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0]
        
    def fetch_intraday_data(self):
        """Fetch intraday SPY data from Polygon.io (SPY tracks SPX closely)"""
        # Get date range for last 15 days (need more to ensure 14+ four-hour periods)
        end_date = datetime.now(self.et_tz).date()
        start_date = end_date - timedelta(days=15)
        
        # Format dates for Polygon API (YYYY-MM-DD)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Polygon.io aggregates endpoint for SPY ETF (tracks SPX)
        url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/{start_str}/{end_str}"
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Print raw response for debugging
            print(f"API Response Status Code: {response.status_code}")
            print(f"API Response Status: {data.get('status', 'No status')}")
            print(f"Result count: {data.get('resultsCount', 0)}")
            
            # Accept both 'OK' and 'DELAYED' status (DELAYED is normal for free tier)
            if data.get('status') not in ['OK', 'DELAYED']:
                error_msg = data.get('error', data.get('message', f"Unexpected status: {data.get('status')}"))
                raise Exception(f"API Error: {error_msg}")
            
            # Convert Polygon format to our expected format
            polygon_data = {}
            if 'results' in data and data['results']:
                for bar in data['results']:
                    # Convert timestamp from milliseconds to datetime
                    dt = datetime.fromtimestamp(bar['t'] / 1000, tz=self.et_tz)
                    timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    polygon_data[timestamp_str] = {
                        'open': float(bar['o']),
                        'high': float(bar['h']),
                        'low': float(bar['l']),
                        'close': float(bar['c']),
                        'volume': int(bar['v'])
                    }
                
                print(f"Successfully fetched {len(polygon_data)} data points")
            else:
                print("No results in API response")
            
            return polygon_data
        
        except Exception as e:
            print(f"Error fetching data: {e}")
            return {}
    
    def load_existing_data(self):
        """Load existing data from JSON file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {"4h_candles": [], "atr_values": [], "current_levels": {}}
    
    def save_data(self, data):
        """Save data to JSON file"""
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def convert_to_4h_candles(self, intraday_data):
        """Convert 5-minute data to 4-hour candles (9:00-13:00 and 13:00-close)"""
        if not intraday_data:
            return []
            
        # Convert to DataFrame for easier manipulation
        df_data = []
        for timestamp_str, values in intraday_data.items():
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            dt = self.et_tz.localize(dt) if dt.tzinfo is None else dt
            
            df_data.append({
                'timestamp': dt,
                'open': values['open'],
                'high': values['high'],
                'low': values['low'],
                'close': values['close'],
                'volume': values['volume']
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp')
        
        # Filter for market hours (9:00 AM - 4:00 PM ET)
        df = df[
            (df['timestamp'].dt.hour >= 9) & 
            (df['timestamp'].dt.hour < 16)
        ]
        
        # Create 4-hour blocks
        four_hour_candles = []
        
        # Group by date first
        for date, day_data in df.groupby(df['timestamp'].dt.date):
            day_data = day_data.sort_values('timestamp')
            
            # Morning block: 9:00 AM - 1:00 PM ET (9:00-13:00)
            morning = day_data[
                (day_data['timestamp'].dt.hour >= 9) & 
                (day_data['timestamp'].dt.hour < 13)
            ]
            
            # Afternoon block: 1:00 PM - close (13:00-16:00)
            afternoon = day_data[
                (day_data['timestamp'].dt.hour >= 13)
            ]
            
            # Create candles for each block
            blocks = [
                ('morning', morning, f"{date} 09:00-13:00"),
                ('afternoon', afternoon, f"{date} 13:00-16:00")
            ]
            
            for block_name, block_data, timestamp_label in blocks:
                if len(block_data) > 0:
                    candle = {
                        'timestamp': timestamp_label,
                        'date': str(date),
                        'period': block_name,
                        'open': block_data.iloc[0]['open'],
                        'high': block_data['high'].max(),
                        'low': block_data['low'].min(),
                        'close': block_data.iloc[-1]['close'],
                        'volume': block_data['volume'].sum()
                    }
                    four_hour_candles.append(candle)
        
        return four_hour_candles
    
    def calculate_true_range(self, candles):
        """Calculate True Range for each 4H candle"""
        if len(candles) < 2:
            return []
            
        tr_values = []
        
        for i in range(1, len(candles)):
            current = candles[i]
            previous = candles[i-1]
            
            # True Range = max of:
            # 1. High - Low
            # 2. |High - Previous Close|
            # 3. |Low - Previous Close|
            tr1 = current['high'] - current['low']
            tr2 = abs(current['high'] - previous['close'])
            tr3 = abs(current['low'] - previous['close'])
            
            true_range = max(tr1, tr2, tr3)
            tr_values.append({
                'timestamp': current['timestamp'],
                'date': current['date'],
                'period': current['period'],
                'true_range': true_range,
                'close': current['close']
            })
        
        return tr_values
    
    def calculate_wilders_atr(self, tr_values, period=14):
        """Calculate ATR using Wilder's smoothing method"""
        if len(tr_values) < period:
            return []
            
        atr_values = []
        
        # First ATR is simple average of first 14 periods
        first_atr = sum(tr['true_range'] for tr in tr_values[:period]) / period
        atr_values.append({
            'timestamp': tr_values[period-1]['timestamp'],
            'date': tr_values[period-1]['date'],
            'period': tr_values[period-1]['period'],
            'atr': first_atr,
            'close': tr_values[period-1]['close']
        })
        
        # Subsequent ATRs use Wilder's smoothing
        # ATR = ((period-1) * previous_ATR + current_TR) / period
        for i in range(period, len(tr_values)):
            prev_atr = atr_values[-1]['atr']
            current_tr = tr_values[i]['true_range']
            
            new_atr = ((period - 1) * prev_atr + current_tr) / period
            
            atr_values.append({
                'timestamp': tr_values[i]['timestamp'],
                'date': tr_values[i]['date'],
                'period': tr_values[i]['period'],
                'atr': new_atr,
                'close': tr_values[i]['close']
            })
        
        return atr_values
    
    def calculate_levels(self, prior_4h_close, current_atr):
        """Calculate Fibonacci-based levels using prior 4H close and current ATR"""
        levels = {}
        
        for ratio in self.fib_ratios:
            # Positive levels (above prior close)
            if ratio == 0.0:
                levels[f"Level_0"] = prior_4h_close
            else:
                levels[f"Level_+{ratio}"] = prior_4h_close + (current_atr * ratio)
                levels[f"Level_-{ratio}"] = prior_4h_close - (current_atr * ratio)
        
        return levels
    
    def get_update_type(self):
        """Determine if this is 1PM or 4PM update based on current time"""
        current_time = datetime.now(self.et_tz)
        current_hour = current_time.hour
        
        # 1PM update uses morning 4H candle (9:00-13:00)
        # 4PM update uses afternoon 4H candle (13:00-16:00)
        if current_hour == 13:  # 1PM ET
            return "1PM_update", "morning"
        else:  # 4PM ET or manual run
            return "4PM_update", "afternoon"
    
    def update_levels(self):
        """Main function to update levels"""
        print(f"Starting levels update at {datetime.now(self.et_tz)}")
        
        # Determine update type
        update_type, target_period = self.get_update_type()
        print(f"Update type: {update_type}, Target period: {target_period}")
        
        # Fetch new intraday data
        intraday_data = self.fetch_intraday_data()
        if not intraday_data:
            print("No new data fetched")
            return
        
        # Load existing data
        stored_data = self.load_existing_data()
        
        # Convert to 4-hour candles
        new_4h_candles = self.convert_to_4h_candles(intraday_data)
        
        # Merge with existing candles (remove duplicates)
        existing_timestamps = {candle['timestamp'] for candle in stored_data['4h_candles']}
        unique_new_candles = [c for c in new_4h_candles if c['timestamp'] not in existing_timestamps]
        
        all_candles = stored_data['4h_candles'] + unique_new_candles
        
        # Sort by timestamp and keep last 30 candles (enough for ATR calculation)
        all_candles = sorted(all_candles, key=lambda x: x['timestamp'])[-30:]
        
        # Calculate True Range
        tr_values = self.calculate_true_range(all_candles)
        
        # Calculate ATR(14) using Wilder's method
        atr_values = self.calculate_wilders_atr(tr_values, period=14)
        
        if not atr_values:
            print("Not enough data for ATR calculation (need at least 14 periods)")
            return
        
        # Get the most recent ATR value
        current_atr_data = atr_values[-1]
        current_atr = current_atr_data['atr']
        
        # Find the prior 4H close for level calculation
        # For 1PM update: use the most recent morning candle's close
        # For 4PM update: use the most recent afternoon candle's close
        prior_4h_close = None
        
        # Look for the most recent candle of the target period
        for candle in reversed(all_candles):
            if candle['period'] == target_period:
                prior_4h_close = candle['close']
                print(f"Using {target_period} close: {prior_4h_close}")
                break
        
        if prior_4h_close is None:
            print(f"Could not find recent {target_period} candle for level calculation")
            return
        
        # Calculate new levels
        new_levels = self.calculate_levels(prior_4h_close, current_atr)
        
        # Update stored data
        stored_data['4h_candles'] = all_candles
        stored_data['atr_values'] = atr_values
        stored_data['current_levels'] = {
            'update_type': update_type,
            'timestamp': datetime.now(self.et_tz).isoformat(),
            'prior_4h_close': prior_4h_close,
            'current_atr': current_atr,
            'levels': new_levels
        }
        
        # Save data
        self.save_data(stored_data)
        
        # Print results
        print(f"\n=== SPY {update_type} LEVELS UPDATE ===")
        print(f"Prior 4H Close ({target_period}): {prior_4h_close:.2f}")
        print(f"Current ATR(14): {current_atr:.2f}")
        print(f"\nCalculated Levels:")
        
        # Sort levels for display
        sorted_levels = sorted(new_levels.items(), key=lambda x: x[1], reverse=True)
        for level_name, level_value in sorted_levels:
            print(f"{level_name}: {level_value:.2f}")
        
        print(f"\nTotal 4H candles stored: {len(all_candles)}")
        print(f"ATR calculation periods: {len(atr_values)}")

def main():
    # Get API key from environment variable
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")
    
    calculator = SPYLevelsCalculator(api_key)
    calculator.update_levels()

if __name__ == "__main__":
    main()
