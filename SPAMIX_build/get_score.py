"""
SPAMIX anomaly score calculator
"""

import numpy as np
import datetime, os, re
import json
import random
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor, as_completed
import tldextract
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
from collections import defaultdict
import pandas as pd


class DNSBL_IP_Feature_Calculator:
    """Calculate DNSBL features for IP addresses"""

    DEFAULT_CONFIG = {
        'dnsbl_list': [
            "spamhaus.org", "spamcop.net", "barracudacentral.org",
            "mailspike.net", "manitu.net", "uceprotect.net"
        ],
        'feature_params': {
            'reputation': {
                'lambda_decay': 0.1,
                'min_records': 10
            },
            'flapping': {
                'center': 10,
                'weight': 0.5,
                'lambda_decay': 0.1,
                'min_records': 10,
                'interval_days': 30,
                'lookback_days': 180,
            },
            'activity_spike': {
                'center': 5,
                'weight': 0.5,
                'lambda_decay': 0.1,
                'min_records': 30,
                'short_window': 7,
                'long_window': 90
            },
            'active_day': {
                'min_cnt_threshold': 1,
                'center': 0.6,
                'weight': 0.5,
                'lookback_days': 180,
                'min_records': 10
            }
        }
    }

    def __init__(self, config=None, data_dir: str = "./data"):
        self.config = config if config else self.DEFAULT_CONFIG
        self.data_dir = data_dir
        self._ip_records_cache = {}

    def _load_ip_records(self, ip: str, dnsbl: str) -> Optional[pd.DataFrame]:
        """Load DNSBL records for an IP from CSV file"""
        cache_key = f"{ip}_{dnsbl}"
        if cache_key in self._ip_records_cache:
            return self._ip_records_cache[cache_key]

        file_path = os.path.join(self.data_dir, "dnsbl_ip_records.csv")
        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_csv(file_path)
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            df['date'] = pd.to_datetime(df['date'])

            filtered = df[(df['ip'] == ip) & (df['dnsbl'] == dnsbl)].sort_values('start_time')
            self._ip_records_cache[cache_key] = filtered if len(filtered) > 0 else None
            return self._ip_records_cache[cache_key]
        except Exception as e:
            print(f"[ERROR] Failed to load IP records for {ip}@{dnsbl}: {e}")
            return None

    @staticmethod
    def sigmoid(x, center, weight):
        """Sigmoid function for normalizing anomaly scores"""
        return 1 / (1 + math.exp(-weight * (x - center)))

    def calculate_features(self, ip, detect_time_str, dnsbl=None, features=None):
        """
        Calculate DNSBL features for a given IP

        Args:
            ip: IP address to query
            detect_time_str: Detection time string, format "%Y-%m-%d %H:%M:%S"
            dnsbl: Specific DNSBL service, if None compute for all
            features: List of features to compute, if None compute all
        Returns:
            Dictionary of feature scores
        """
        detect_time = datetime.strptime(detect_time_str, "%Y-%m-%d %H:%M:%S")
        results = {}

        dnsbls = [dnsbl] if dnsbl else self.config['dnsbl_list']
        all_features = ['reputation', 'flapping', 'activity_spike', 'active_day']
        features_to_calc = features if features else all_features

        for dnsbl_service in dnsbls:
            dnsbl_result = {}
            records_df = self._load_ip_records(ip, dnsbl_service)

            if records_df is None:
                results[dnsbl_service] = {f: None for f in features_to_calc}
                continue

            if 'reputation' in features_to_calc:
                dnsbl_result['reputation'] = self._calculate_reputation(records_df, detect_time)

            if 'flapping' in features_to_calc:
                dnsbl_result['flapping'] = self._calculate_flapping(records_df, detect_time)

            if 'activity_spike' in features_to_calc:
                daily_stats = self._get_daily_stats(records_df, detect_time)
                dnsbl_result['activity_spike'] = self._calculate_activity_spike(daily_stats, detect_time)

            if 'active_day' in features_to_calc:
                daily_stats = self._get_daily_stats(records_df, detect_time)
                dnsbl_result['active_day'] = self._calculate_active_day(daily_stats)

            results[dnsbl_service] = dnsbl_result

        return results

    def _get_daily_stats(self, records_df: pd.DataFrame, detect_time: datetime) -> Optional[pd.DataFrame]:
        """Get daily aggregated statistics"""
        if records_df is None or len(records_df) == 0:
            return None

        params = self.config['feature_params']['activity_spike']
        start_time_limit = detect_time - relativedelta(months=12)

        records_df = records_df[records_df['start_time'] >= start_time_limit]
        records_df = records_df[records_df['end_time'] <= detect_time]

        if len(records_df) < params['min_records']:
            return None

        daily_agg = records_df.groupby(records_df['date'].dt.date)['cnt'].sum().reset_index()
        daily_agg.columns = ['date', 'cnt']
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        daily_agg = daily_agg.set_index('date').sort_index()

        full_range = pd.date_range(start=start_time_limit.date(), end=detect_time.date() - timedelta(days=1))
        return daily_agg.reindex(full_range, fill_value=0)

    def _calculate_reputation(self, records_df: pd.DataFrame, detect_time: datetime) -> Optional[float]:
        """Calculate DNSBL reputation score"""
        params = self.config['feature_params']['reputation']
        rep_list_sum = 0.0
        rep_delist_sum = 0.0

        if records_df is None or len(records_df) < params['min_records']:
            return None

        for _, row in records_df.iterrows():
            flag = row['flag']
            start_time = row['start_time']
            end_time = row['end_time']
            cnt = row['cnt']
            client_count = row['client_count']

            D = (end_time - start_time).total_seconds() / 86400
            delta_t = (detect_time - start_time).total_seconds() / 86400

            if D <= 0 or delta_t < 0:
                continue

            weight = math.exp(-params['lambda_decay'] * delta_t)
            cnt_ln = math.log(cnt + 1)
            client_ln = math.log(client_count + 1)
            score = weight * D * cnt_ln * client_ln

            if flag == "black":
                rep_list_sum += score
            elif flag == "white":
                rep_delist_sum += score

        total = rep_list_sum + rep_delist_sum
        return rep_list_sum / total if total > 0 else 0.0

    def _calculate_flapping(self, records_df: pd.DataFrame, detect_time: datetime) -> Optional[float]:
        """Calculate DNSBL flapping behavior score"""
        if records_df is None or len(records_df) == 0:
            return None

        params = self.config['feature_params']['flapping']
        interval_records = defaultdict(list)
        current_interval_start = detect_time - relativedelta(days=params['lookback_days'])

        for _, row in records_df.iterrows():
            flag = row['flag']
            start_time = row['start_time']
            end_time = row['end_time']

            while start_time >= current_interval_start + relativedelta(days=params['interval_days']):
                current_interval_start += relativedelta(days=params['interval_days'])

            interval_key = current_interval_start.strftime("%Y-%m-%d")
            interval_records[interval_key].append((flag, start_time, end_time))

        total_score = 0.0
        total_weight = 0.0

        for interval_str, recs in interval_records.items():
            flags = [r[0] for r in recs]
            if len(flags) < 2:
                continue

            if all(f == "black" for f in flags):
                flap_count = len(flags)
            else:
                flap_count = sum(1 for i in range(1, len(flags)) if flags[i] != flags[i - 1])

            flap_score = self.sigmoid(flap_count, center=params['center'], weight=params['weight'])

            interval_start = datetime.strptime(interval_str, "%Y-%m-%d")
            interval_mid_time = interval_start + relativedelta(days=params['interval_days'] / 2)
            delta_t = (detect_time - interval_mid_time).days

            if delta_t >= 0:
                decay_weight = math.exp(-params['lambda_decay'] * delta_t)
                total_score += flap_score * decay_weight
                total_weight += decay_weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_activity_spike(self, daily_stats: Optional[pd.DataFrame], detect_time: datetime) -> Optional[float]:
        """Calculate short-term vs long-term traffic spike anomaly score (7d vs 90d)"""
        if daily_stats is None:
            return None

        params = self.config['feature_params']['activity_spike']
        short_w = params['short_window']
        long_w = params['long_window']

        if len(daily_stats) < long_w:
            return None

        total_score = 0.0
        total_weight = 0.0

        for i in range(long_w, len(daily_stats)):
            long_window = daily_stats.iloc[i - long_w:i]
            long_avg = long_window['cnt'].mean()
            short_window = daily_stats.iloc[i - short_w:i]
            short_avg = short_window['cnt'].mean()
            current_day = daily_stats.index[i]

            if long_avg < 50 and short_avg < 50:
                continue

            if long_avg == 0 and short_avg == 0:
                continue
            elif long_avg == 0 and short_avg > 0:
                change_ratio = float('inf')
            else:
                change_ratio = short_avg / long_avg
                if change_ratio <= 0:
                    continue

            spike_score = 1.0 if change_ratio == float('inf') else \
                self.sigmoid(change_ratio, center=params['center'], weight=params['weight'])

            delta_t = (detect_time - current_day).days
            if delta_t < 0:
                continue

            decay_weight = math.exp(-params['lambda_decay'] * delta_t)
            total_score += spike_score * decay_weight
            total_weight += decay_weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_active_day(self, daily_stats: Optional[pd.DataFrame]) -> Optional[float]:
        """Calculate query activity day ratio"""
        if daily_stats is None or len(daily_stats) < 30:
            return None

        params = self.config['feature_params']['active_day']
        lookback_days = min(params['lookback_days'], len(daily_stats))

        recent_stats = daily_stats.iloc[-lookback_days:]
        active_days = (recent_stats['cnt'] >= params['min_cnt_threshold']).sum()
        active_ratio = (lookback_days - active_days) / lookback_days

        return self.sigmoid(active_ratio, center=params['center'], weight=params['weight'])


class DNSBL_SLD_Feature_Calculator:
    """Calculate DNSBL features for SLD (Second Level Domain)"""

    DEFAULT_CONFIG = {
        'dnsbl_list': ["spamhaus.org", "surbl.org"],
        'feature_params': {
            'reputation': {
                'lambda_decay': 0.1,
                'min_records': 10
            },
            'flapping': {
                'center': 5,
                'weight': 0.5,
                'lambda_decay': 0.1,
                'min_records': 10,
                'interval_days': 30,
                'lookback_days': 360,
            },
            'activity_spike': {
                'center': 5,
                'weight': 0.5,
                'lambda_decay': 0.1,
                'min_records': 30,
                'short_window': 7,
                'long_window': 90
            },
            'active_day': {
                'min_cnt_threshold': 1,
                'center': 0.6,
                'weight': 0.5,
                'lookback_days': 180,
                'min_records': 10
            }
        }
    }

    def __init__(self, config=None, data_dir: str = "./data"):
        self.config = config if config else self.DEFAULT_CONFIG
        self.data_dir = data_dir
        self._sld_records_cache = {}

    def _load_sld_records(self, sld: str, dnsbl: str) -> Optional[pd.DataFrame]:
        """Load DNSBL records for an SLD from CSV file"""
        cache_key = f"{sld}_{dnsbl}"
        if cache_key in self._sld_records_cache:
            return self._sld_records_cache[cache_key]

        file_path = os.path.join(self.data_dir, "dnsbl_sld_records.csv")
        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_csv(file_path)
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            df['date'] = pd.to_datetime(df['date'])

            filtered = df[(df['sld'] == sld) & (df['dnsbl'] == dnsbl)].sort_values('start_time')
            self._sld_records_cache[cache_key] = filtered if len(filtered) > 0 else None
            return self._sld_records_cache[cache_key]
        except Exception as e:
            print(f"[ERROR] Failed to load SLD records for {sld}@{dnsbl}: {e}")
            return None

    @staticmethod
    def sigmoid(x, center, weight):
        return 1 / (1 + math.exp(-weight * (x - center)))

    def calculate_features(self, sld, detect_time_str, dnsbl=None, features=None):
        """Calculate DNSBL features for a given SLD"""
        detect_time = datetime.strptime(detect_time_str, "%Y-%m-%d %H:%M:%S")
        results = {}

        dnsbls = [dnsbl] if dnsbl else self.config['dnsbl_list']
        all_features = ['reputation', 'flapping', 'activity_spike', 'active_day']
        features_to_calc = features if features else all_features

        for dnsbl_service in dnsbls:
            dnsbl_result = {}
            records_df = self._load_sld_records(sld, dnsbl_service)

            if records_df is None:
                results[dnsbl_service] = {f: None for f in features_to_calc}
                continue

            if 'reputation' in features_to_calc:
                dnsbl_result['reputation'] = self._calculate_reputation(records_df, detect_time)

            if 'flapping' in features_to_calc:
                dnsbl_result['flapping'] = self._calculate_flapping(records_df, detect_time)

            if 'activity_spike' in features_to_calc:
                daily_stats = self._get_daily_stats(records_df, detect_time)
                dnsbl_result['activity_spike'] = self._calculate_activity_spike(daily_stats, detect_time)

            if 'active_day' in features_to_calc:
                daily_stats = self._get_daily_stats(records_df, detect_time)
                dnsbl_result['active_day'] = self._calculate_active_day(daily_stats)

            results[dnsbl_service] = dnsbl_result

        return results

    def _get_daily_stats(self, records_df: pd.DataFrame, detect_time: datetime) -> Optional[pd.DataFrame]:
        """Get daily aggregated statistics"""
        if records_df is None or len(records_df) == 0:
            return None

        params = self.config['feature_params']['activity_spike']
        start_time_limit = detect_time - relativedelta(months=12)

        records_df = records_df[records_df['start_time'] >= start_time_limit]
        records_df = records_df[records_df['end_time'] <= detect_time]

        if len(records_df) < params['min_records']:
            return None

        daily_agg = records_df.groupby(records_df['date'].dt.date)['cnt'].sum().reset_index()
        daily_agg.columns = ['date', 'cnt']
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        daily_agg = daily_agg.set_index('date').sort_index()

        full_range = pd.date_range(start=start_time_limit.date(), end=detect_time.date() - timedelta(days=1))
        return daily_agg.reindex(full_range, fill_value=0)

    def _calculate_reputation(self, records_df: pd.DataFrame, detect_time: datetime) -> Optional[float]:
        """Calculate DNSBL reputation score for SLD"""
        params = self.config['feature_params']['reputation']
        rep_list_sum = 0.0
        rep_delist_sum = 0.0

        if records_df is None or len(records_df) < params['min_records']:
            return None

        for _, row in records_df.iterrows():
            flag = row['flag']
            start_time = row['start_time']
            end_time = row['end_time']
            cnt = row['cnt']
            client_count = row['client_count']

            D = (end_time - start_time).total_seconds() / 86400
            delta_t = (detect_time - start_time).total_seconds() / 86400

            if D <= 0 or delta_t < 0:
                continue

            weight = math.exp(-params['lambda_decay'] * delta_t)
            score = weight * D * math.log(cnt + 1) * math.log(client_count + 1)

            if flag == "black":
                rep_list_sum += score
            elif flag == "white":
                rep_delist_sum += score

        total = rep_list_sum + rep_delist_sum
        return rep_list_sum / total if total > 0 else 0.0

    def _calculate_flapping(self, records_df: pd.DataFrame, detect_time: datetime) -> Optional[float]:
        """Calculate flapping behavior score for SLD"""
        if records_df is None or len(records_df) == 0:
            return None

        params = self.config['feature_params']['flapping']
        interval_records = defaultdict(list)
        current_interval_start = detect_time - relativedelta(days=params['lookback_days'])

        for _, row in records_df.iterrows():
            flag = row['flag']
            start_time = row['start_time']
            end_time = row['end_time']

            while start_time >= current_interval_start + relativedelta(days=params['interval_days']):
                current_interval_start += relativedelta(days=params['interval_days'])

            interval_key = current_interval_start.strftime("%Y-%m-%d")
            interval_records[interval_key].append((flag, start_time, end_time))

        total_score = 0.0
        total_weight = 0.0

        for interval_str, recs in interval_records.items():
            flags = [r[0] for r in recs]
            if len(flags) < 2:
                continue

            if all(f == "black" for f in flags):
                flap_count = len(flags)
            else:
                flap_count = sum(1 for i in range(1, len(flags)) if flags[i] != flags[i - 1])

            flap_score = self.sigmoid(flap_count, center=params['center'], weight=params['weight'])

            interval_start = datetime.strptime(interval_str, "%Y-%m-%d")
            interval_mid_time = interval_start + relativedelta(days=params['interval_days'] / 2)
            delta_t = (detect_time - interval_mid_time).days

            if delta_t >= 0:
                decay_weight = math.exp(-params['lambda_decay'] * delta_t)
                total_score += flap_score * decay_weight
                total_weight += decay_weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_activity_spike(self, daily_stats: Optional[pd.DataFrame], detect_time: datetime) -> Optional[float]:
        """Calculate activity spike score (7d vs 90d)"""
        if daily_stats is None:
            return None

        params = self.config['feature_params']['activity_spike']
        short_w = params['short_window']
        long_w = params['long_window']

        if len(daily_stats) < long_w:
            return None

        total_score = 0.0
        total_weight = 0.0

        for i in range(long_w, len(daily_stats)):
            long_avg = daily_stats.iloc[i - long_w:i]['cnt'].mean()
            short_avg = daily_stats.iloc[i - short_w:i]['cnt'].mean()
            current_day = daily_stats.index[i]

            if long_avg < 50 and short_avg < 50:
                continue

            if long_avg == 0 and short_avg == 0:
                continue
            elif long_avg == 0:
                change_ratio = float('inf')
            else:
                change_ratio = short_avg / long_avg
                if change_ratio <= 0:
                    continue

            spike_score = 1.0 if change_ratio == float('inf') else \
                self.sigmoid(change_ratio, center=params['center'], weight=params['weight'])

            delta_t = (detect_time - current_day).days
            if delta_t < 0:
                continue

            decay_weight = math.exp(-params['lambda_decay'] * delta_t)
            total_score += spike_score * decay_weight
            total_weight += decay_weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_active_day(self, daily_stats: Optional[pd.DataFrame]) -> Optional[float]:
        """Calculate active day ratio"""
        if daily_stats is None or len(daily_stats) < 30:
            return None

        params = self.config['feature_params']['active_day']
        lookback_days = min(params['lookback_days'], len(daily_stats))

        recent_stats = daily_stats.iloc[-lookback_days:]
        active_days = (recent_stats['cnt'] >= params['min_cnt_threshold']).sum()
        active_ratio = (lookback_days - active_days) / lookback_days

        return self.sigmoid(active_ratio, center=params['center'], weight=params['weight'])


class Email_Feature_Calculator:
    """Calculate email-based features for spam detection"""

    DEFAULT_CONFIG = {
        'feature_params': {
            'reputation': {
                'lambda_decay': 0.1,
                'min_total_emails': 50,
                'min_records': 10,
                'lookback_days': 365
            },
            'active_score': {
                'min_total_emails': 50,
                'center': 0.6,
                'weight': 10,
                'min_records': 10,
                'max_window_days': 180,
                'min_window_days': 30
            },
            'dkim_pass': {
                'min_total_emails': 50,
                'center': 0.5,
                'weight': 10,
                'lambda_decay': 0.1,
                'min_records': 10,
                'lookback_days': 365
            },
            'activity_spike': {
                'center': 5,
                'weight': 0.5,
                'lambda_decay': 0.1,
                'min_total_emails': 50,
                'min_records': 30,
                'short_window': 7,
                'long_window': 90
            }
        }
    }

    def __init__(self, config=None, data_dir: str = "./data"):
        self.config = config if config else self.DEFAULT_CONFIG
        self.data_dir = data_dir
        self._email_cache = {}

    def _load_email_data(self, target: str) -> Optional[pd.DataFrame]:
        """Load email reputation data for a target from CSV file"""
        if target in self._email_cache:
            return self._email_cache[target]

        file_path = os.path.join(self.data_dir, "email_reputation.csv")
        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])

            filtered = df[df['target'] == target].sort_values('date')
            if len(filtered) > 0:
                self._email_cache[target] = filtered
                return filtered
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load email data for {target}: {e}")
            return None

    @staticmethod
    def sigmoid(x, center, weight):
        return 1 / (1 + math.exp(-weight * (x - center)))

    def calculate_features(self, target, detect_time_str, features=None):
        """Calculate email-based features for a given target (ip##domain)"""
        detect_time = pd.to_datetime(detect_time_str)
        results = {}

        all_features = ['reputation', 'active_score', 'dkim_pass', 'activity_spike']
        features_to_calc = features if features else all_features

        email_data = self._load_email_data(target)

        if email_data is None:
            return {f: None for f in features_to_calc}

        if 'reputation' in features_to_calc:
            results['reputation'] = self._calculate_reputation(email_data, detect_time)

        if 'active_score' in features_to_calc:
            results['active_score'] = self._calculate_active_score(email_data, detect_time)

        if 'dkim_pass' in features_to_calc:
            results['dkim_pass'] = self._calculate_dkim_score(email_data, detect_time)

        if 'activity_spike' in features_to_calc:
            results['activity_spike'] = self._calculate_activity_spike(email_data, detect_time)

        return results

    def _calculate_activity_spike(self, email_data: pd.DataFrame, detect_time: pd.Timestamp) -> Optional[float]:
        """Calculate email traffic spike (7d vs 90d)"""
        if email_data is None or len(email_data) == 0:
            return None

        params = self.config['feature_params']['activity_spike']
        short_w = params['short_window']
        long_w = params['long_window']

        df = email_data.copy()
        df['total_cnt'] = df['spam_cnt'] + df['normal_cnt']
        df = df.sort_values('date').reset_index(drop=True)

        if len(df) < long_w:
            return None

        total_score = 0.0
        total_weight = 0.0

        for i in range(long_w, len(df)):
            long_avg = df.iloc[i - long_w:i]['total_cnt'].mean()
            short_avg = df.iloc[i - short_w:i]['total_cnt'].mean()
            current_day = df.iloc[i]['date']

            if long_avg < params['min_total_emails'] and short_avg < params['min_total_emails']:
                continue

            if long_avg == 0 and short_avg == 0:
                continue
            elif long_avg == 0:
                change_ratio = float('inf')
            else:
                change_ratio = short_avg / long_avg
                if change_ratio <= 0:
                    continue

            spike_score = 1.0 if change_ratio == float('inf') else \
                self.sigmoid(change_ratio, center=params['center'], weight=params['weight'])

            delta_t = (detect_time - current_day).days
            if delta_t < 0:
                continue

            decay_weight = math.exp(-params['lambda_decay'] * delta_t)
            total_score += spike_score * decay_weight
            total_weight += decay_weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_reputation(self, email_data: pd.DataFrame, detect_time: pd.Timestamp) -> Optional[float]:
        """Calculate email reputation score"""
        params = self.config['feature_params']['reputation']

        email_data = email_data.copy()
        email_data['total_cnt'] = email_data['spam_cnt'] + email_data['normal_cnt']
        email_data = email_data[email_data['total_cnt'] >= params['min_total_emails']]

        if len(email_data) < params['min_records']:
            return None

        email_data = email_data.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
        email_data['delta_days'] = (detect_time - email_data['date']).dt.days
        weights = np.exp(-params['lambda_decay'] * email_data['delta_days'].to_numpy())
        spam_ratios = (email_data['spam_cnt'] / email_data['total_cnt']).to_numpy()

        weighted_sum = np.sum(spam_ratios * weights)
        total_weight = np.sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else None

    def _calculate_active_score(self, email_data: pd.DataFrame, detect_time: pd.Timestamp) -> Optional[float]:
        """Calculate email activity anomaly score"""
        if email_data is None or len(email_data) == 0:
            return None

        params = self.config['feature_params']['active_score']

        df = email_data.copy()
        df['total_cnt'] = df['spam_cnt'] + df['normal_cnt']
        df = df[df['total_cnt'] >= params['min_total_emails']]

        if len(df) < params['min_records']:
            return None

        first_date = df['date'].min()
        interval_days = (detect_time - first_date).days

        if interval_days < params['min_window_days']:
            return None

        window_days = min(interval_days, params['max_window_days'])
        start_time = detect_time - timedelta(days=window_days)

        df_window = df[df['date'] >= start_time]

        full_dates = pd.date_range(start=start_time, end=detect_time - timedelta(days=1))
        active_dates = set(df_window['date'].dt.date)

        active_days = sum(1 for date in full_dates if date.date() in active_dates)
        active_ratio = active_days / window_days

        return self.sigmoid(1 - active_ratio, center=params['center'], weight=params['weight'])

    def _calculate_dkim_score(self, email_data: pd.DataFrame, detect_time: pd.Timestamp) -> Optional[float]:
        """Calculate DKIM pass rate anomaly score"""
        return self._calculate_auth_score(email_data, detect_time, 'dkim_pass')

    def _calculate_auth_score(self, email_data: pd.DataFrame, detect_time: pd.Timestamp, auth_type: str) -> Optional[
        float]:
        """Generic authentication score calculation (SPF/DKIM)"""
        params = self.config['feature_params'][auth_type]

        df = email_data.copy()
        df['total_cnt'] = df['spam_cnt'] + df['normal_cnt']
        df = df[df['total_cnt'] >= params['min_total_emails']]

        if len(df) < params['min_records']:
            return None

        df[f'{auth_type}_ratio'] = (df[f'spam_{auth_type}'] + df[f'normal_{auth_type}']) / df['total_cnt']
        df['norm_score'] = df[f'{auth_type}_ratio'].apply(
            lambda x: self.sigmoid(1 - x, center=params['center'], weight=params['weight']))

        df['days_since'] = (detect_time - df['date']).dt.days
        df['decay_weight'] = np.exp(-params['lambda_decay'] * df['days_since'])

        numerator = (df['norm_score'] * df['decay_weight']).sum()
        denominator = df['decay_weight'].sum()

        return numerator / denominator if denominator > 0 else 0.0


# ===== Feature Weight Configurations =====
FEATURE_WEIGHT_SETS = {
    "lr_model": {
        "email_active_score": 0.02545021257186454,
        "email_activity_spike": 0.001097647635083565,
        "email_dkim_pass": 0.0149773046443811,
        "email_reputation": 0.21722845574509092,
        "ip_barracudacentral.org_active_day": 0.018464160599395027,
        "ip_barracudacentral.org_activity_spike": 0.006102381218079755,
        "ip_barracudacentral.org_flapping": 0.0035130557800019903,
        "ip_barracudacentral.org_reputation": 0.0054094546358870255,
        "ip_mailspike.net_active_day": 0.024496217903430973,
        "ip_mailspike.net_activity_spike": 0.00111495455186944113,
        "ip_mailspike.net_flapping": 0.071120107457302663,
        "ip_mailspike.net_reputation": 0.002570872012822638,
        "ip_spamcop.net_active_day": 0.010508108991185094,
        "ip_spamcop.net_activity_spike": 0.002624142766890137,
        "ip_spamcop.net_flapping": 0.015030883045764635,
        "ip_spamcop.net_reputation": 0.027040900364594143,
        "ip_spamhaus.org_active_day": 0.05756019066057157,
        "ip_spamhaus.org_activity_spike": 0.027047252424410266,
        "ip_spamhaus.org_flapping": 0.036390631427140094,
        "ip_spamhaus.org_reputation": 0.050343710319899125,
        "ip_uceprotect.net_active_day": 0.02288213514896683,
        "ip_uceprotect.net_activity_spike": 0.008650790471777803,
        "ip_uceprotect.net_flapping": 0.0015366484638523878,
        "ip_uceprotect.net_reputation": 0.0073011685364167715,
        "sld_spamhaus.org_active_day": 0.0029954298121708622,
        "sld_spamhaus.org_activity_spike": 0.005131834728729843,
        "sld_spamhaus.org_flapping": 0.059106653906454025,
        "sld_spamhaus.org_reputation": 0.1835021467043492,
        "sld_surbl.org_active_day": 0.011532672990312132,
        "sld_surbl.org_activity_spike": 0.016838694117603716,
        "sld_surbl.org_flapping": 0.038279150765264,
        "sld_surbl.org_reputation": 0.02415202959843774
    }
}

# ===== DNSBL Configurations =====
DNSBLS_IP = {
    "spamhaus.org": ["reputation", "flapping", "activity_spike", "active_day"],
    "spamcop.net": ["reputation", "flapping", "activity_spike", "active_day"],
    "barracudacentral.org": ["reputation", "flapping", "activity_spike", "active_day"],
    "mailspike.net": ["reputation", "flapping", "activity_spike", "active_day"],
    "manitu.net": ["reputation", "flapping", "activity_spike", "active_day"],
}

DNSBLS_SLD = {
    "spamhaus.org": ["reputation", "flapping", "activity_spike", "active_day"],
    "surbl.org": ["reputation", "flapping", "activity_spike", "active_day"],
}

EMAIL_FEATURES = ["reputation", "active_score", "dkim_pass", "activity_spike"]


# ===== Helper Functions =====
def extract_sld(target: str) -> str:
    """Extract Second Level Domain from target string"""
    ext = tldextract.extract(target)
    if not ext.domain or not ext.suffix:
        return None
    return ext.domain + "." + ext.suffix


def generate_days(start_date: str, end_date: str):
    """Generate list of dates between start_date and end_date"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days)]


def load_targets_from_csv(detect_date: str, data_dir: str = "./data", min_mail_count: int = 500) -> List[Tuple]:
    """
    Load targets (email senders) from CSV file for a given detection date

    CSV format: target, date, spam_cnt, normal_cnt
    """
    file_path = os.path.join(data_dir, "email_targets.csv")
    if not os.path.exists(file_path):
        print(f"[WARNING] Target file not found: {file_path}")
        return []

    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")

        day_data = df[df['date'] == detect_date]
        if day_data.empty:
            return []

        # Aggregate by target
        agg = day_data.groupby('target').agg({
            'spam_cnt': 'sum',
            'normal_cnt': 'sum'
        }).reset_index()

        agg['total_cnt'] = agg['spam_cnt'] + agg['normal_cnt']
        agg = agg[agg['total_cnt'] >= min_mail_count]

        targets = [(row['target'], row['spam_cnt'], row['normal_cnt'], row['total_cnt'])
                   for _, row in agg.iterrows()]
        return targets
    except Exception as e:
        print(f"[ERROR] Failed to load targets for {detect_date}: {e}")
        return []


# ===== Core Scoring Functions =====
def calculate_weighted_scores(feature_values: dict) -> dict:
    """Calculate weighted scores for all weight sets"""
    scores = {}
    for name, weights in FEATURE_WEIGHT_SETS.items():
        score, w_sum = 0.0, 0.0
        for feat, weight in weights.items():
            if feat in feature_values:
                score += feature_values[feat] * weight
                w_sum += weight
        if w_sum < sum(weights.values()) * 0.5:
            scores[f"score_{name}"] = -1
        else:
            scores[f"score_{name}"] = score / w_sum
    return scores


def calculate_daily_score(target: str, detect_time: str, mail_count: int,
                          spam_cnt: int, normal_cnt: int, data_dir: str = "./data") -> dict:
    """Calculate all features and scores for a single target on a specific date"""
    feature_values = {}

    try:
        ip, domain = target.split("##")
    except ValueError:
        return {"target": target, "date": detect_time, "score_baseline": -1,
                "mail_count": mail_count, "spam_cnt": spam_cnt, "normal_cnt": normal_cnt}

    detect_time_full = f"{detect_time} 00:00:00"
    sld = extract_sld(domain)

    # 1. DNSBL IP Features
    try:
        dnsbl_ip_calc = DNSBL_IP_Feature_Calculator(data_dir=data_dir)
        for dnsbl, feats in DNSBLS_IP.items():
            dnsbl_result = dnsbl_ip_calc.calculate_features(ip, detect_time_full, dnsbl=dnsbl)
            if isinstance(dnsbl_result, dict) and dnsbl in dnsbl_result:
                for f, v in dnsbl_result[dnsbl].items():
                    if v is not None:
                        feature_values[f"ip_{dnsbl}_{f}"] = v
    except Exception as e:
        pass

    # 2. DNSBL SLD Features
    try:
        dnsbl_sld_calc = DNSBL_SLD_Feature_Calculator(data_dir=data_dir)
        for dnsbl, feats in DNSBLS_SLD.items():
            if sld:
                dnsbl_result = dnsbl_sld_calc.calculate_features(sld, detect_time_full, dnsbl=dnsbl)
                if isinstance(dnsbl_result, dict) and dnsbl in dnsbl_result:
                    for f, v in dnsbl_result[dnsbl].items():
                        if v is not None:
                            feature_values[f"sld_{dnsbl}_{f}"] = v
    except Exception as e:
        pass

    # 3. Email Features
    try:
        email_calc = Email_Feature_Calculator(data_dir=data_dir)
        email_result = email_calc.calculate_features(target, detect_time_full)
        for f in EMAIL_FEATURES:
            v = email_result.get(f)
            if v is not None:
                feature_values[f"email_{f}"] = v
    except Exception as e:
        pass

    # Calculate weighted scores
    score_results = calculate_weighted_scores(feature_values)

    result = {
        "target": target,
        "date": detect_time,
        "mail_count": mail_count,
        "spam_cnt": spam_cnt,
        "normal_cnt": normal_cnt,
        **score_results,
        **feature_values,
    }
    return result


def compute_daily_scores(detect_time: str, records: list, max_workers: int = 40, data_dir: str = "./data"):
    """Compute scores for all targets on a specific day using multi-threading"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(calculate_daily_score, rec[0], detect_time, rec[3], rec[1], rec[2], data_dir)
            for rec in records
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scoring {detect_time[:10]}"):
            try:
                results.append(future.result())
            except Exception:
                continue
    return results


# ===== Main Function =====
def main():
    START_DATE = "2024-01-01"
    END_DATE = "2024-08-01"
    DATA_DIR = "./data"

    detect_times = generate_days(START_DATE, END_DATE)
    all_scores = []

    for detect_time in detect_times:
        targets = load_targets_from_csv(detect_time, data_dir=DATA_DIR, min_mail_count=500)

        if not targets:
            print(f"[{detect_time[:10]}] No valid targets (mail_count>=500).")
            continue

        day_scores = compute_daily_scores(detect_time, targets, data_dir=DATA_DIR)
        all_scores.extend(day_scores)

    # Save results to CSV
    if all_scores:
        df = pd.DataFrame(all_scores)
        score_cols = [f"score_{name}" for name in FEATURE_WEIGHT_SETS.keys()]
        feature_cols = sorted(set().union(*[w.keys() for w in FEATURE_WEIGHT_SETS.values()]))
        fixed_cols = ["target", "date", "mail_count", "spam_cnt", "normal_cnt"] + score_cols + feature_cols
        df = df.reindex(columns=fixed_cols, fill_value=-1)
        df.to_csv("daily_ip_domain_scores_multiweights_2401_2408.csv", index=False)
        print("Multi-weight daily scores saved to daily_ip_domain_scores_multiweights_2401_2408.csv")
    else:
        print("No scores were computed.")


if __name__ == "__main__":
    main()