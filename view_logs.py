#!/usr/bin/env python3
"""
Simple log viewer for ML CSV Analyzer application
Usage: 
  python view_logs.py                      # View user activity logs
  python view_logs.py --analyze            # Show analytics summary
  python view_logs.py --user john_doe      # Filter by user
  python view_logs.py --ip 192.168.1.100  # Filter by IP address
  python view_logs.py --errors-only        # Show only errors
  python view_logs.py --performance        # Show performance metrics
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import argparse

def load_logs(log_file):
    """Load and parse JSON logs from a file"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return []
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line[:100]}... Error: {e}")
    
    return logs

def analyze_logs(logs):
    """Analyze logs and provide summary statistics"""
    if not logs:
        print("No logs to analyze")
        return
    
    print(f"\n📊 ANALYSIS OF {len(logs)} LOG ENTRIES")
    print("=" * 50)
    
    # Time range
    timestamps = [datetime.fromisoformat(log['timestamp']) for log in logs]
    min_time = min(timestamps)
    max_time = max(timestamps)
    print(f"📅 Time Range: {min_time.strftime('%Y-%m-%d %H:%M')} to {max_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"⏱️  Duration: {max_time - min_time}")
    
    # Event types
    event_types = Counter(log.get('event_type', 'unknown') for log in logs)
    print(f"\n🎯 Top Event Types:")
    for event_type, count in event_types.most_common(10):
        print(f"  - {event_type}: {count}")
    
    # Users
    users = Counter(log.get('user_id', 'unknown') for log in logs)
    print(f"\n👥 Active Users ({len(users)} total):")
    for user_id, count in users.most_common(10):
        print(f"  - {user_id}: {count} events")
    
    # IP addresses
    ips = Counter(log.get('client_ip', 'unknown') for log in logs)
    print(f"\n🌐 Unique IP Addresses ({len(ips)} total):")
    for ip, count in ips.most_common(10):
        print(f"  - {ip}: {count} events")
    
    # Log levels
    levels = Counter(log.get('level', 'INFO') for log in logs)
    print(f"\n📊 Log Levels:")
    for level, count in levels.items():
        print(f"  - {level}: {count}")
    
    # Recent activity (last 24 hours)
    recent_cutoff = max_time - timedelta(hours=24)
    recent_logs = [log for log in logs if datetime.fromisoformat(log['timestamp']) > recent_cutoff]
    print(f"\n🕒 Recent Activity (last 24h): {len(recent_logs)} events")

def view_user_activity(logs, user_id=None, ip_address=None):
    """View activity for a specific user, IP, or all users"""
    filtered_logs = logs
    
    if user_id:
        filtered_logs = [log for log in filtered_logs if log.get('user_id') == user_id]
        print(f"\n👤 USER ACTIVITY: {user_id} ({len(filtered_logs)} events)")
        print("-" * 40)
    elif ip_address:
        filtered_logs = [log for log in filtered_logs if log.get('client_ip') == ip_address]
        print(f"\n🌐 IP ACTIVITY: {ip_address} ({len(filtered_logs)} events)")
        print("-" * 40)
    else:
        print(f"\n👥 ALL USER ACTIVITY ({len(filtered_logs)} events)")
        print("-" * 40)
    
    for log in sorted(filtered_logs, key=lambda x: x['timestamp'], reverse=True)[:20]:
        timestamp = datetime.fromisoformat(log['timestamp']).strftime('%m-%d %H:%M')
        event_type = log.get('event_type', 'unknown')
        message = log.get('message', '')
        client_ip = log.get('client_ip', 'unknown')
        print(f"  {timestamp} | {client_ip:<15} | {event_type:<20} | {message}")

def view_errors(logs):
    """View all error logs"""
    error_logs = [log for log in logs if log.get('level') == 'ERROR']
    print(f"\n❌ ERROR LOGS ({len(error_logs)} errors)")
    print("-" * 40)
    
    for log in sorted(error_logs, key=lambda x: x['timestamp'], reverse=True):
        timestamp = datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        event_type = log.get('event_type', 'unknown')
        message = log.get('message', '')
        error_data = log.get('data', {})
        
        print(f"\n🔴 {timestamp} | {event_type}")
        print(f"   Message: {message}")
        if error_data.get('error_type'):
            print(f"   Error: {error_data['error_type']} - {error_data.get('error_message', '')}")
        if error_data.get('user_id'):
            print(f"   User: {error_data['user_id']}")

def view_performance(logs):
    """View performance metrics"""
    perf_logs = [log for log in logs if log.get('data', {}).get('duration_seconds')]
    
    if not perf_logs:
        print("\n⚡ No performance data available")
        return
    
    print(f"\n⚡ PERFORMANCE METRICS ({len(perf_logs)} events)")
    print("-" * 40)
    
    # Group by event type
    perf_by_type = defaultdict(list)
    for log in perf_logs:
        event_type = log.get('event_type', 'unknown')
        duration = log.get('data', {}).get('duration_seconds', 0)
        perf_by_type[event_type].append(duration)
    
    for event_type, durations in perf_by_type.items():
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        print(f"  {event_type}:")
        print(f"    Average: {avg_duration:.2f}s")
        print(f"    Range: {min_duration:.2f}s - {max_duration:.2f}s")
        print(f"    Count: {len(durations)}")
        print()

def main():
    parser = argparse.ArgumentParser(description='View ML CSV Analyzer logs')
    parser.add_argument('--log-type', choices=['app', 'user', 'performance', 'errors'], 
                       default='user', help='Type of log to view')
    parser.add_argument('--user', help='Filter by specific user ID')
    parser.add_argument('--ip', help='Filter by specific IP address')
    parser.add_argument('--analyze', action='store_true', help='Show analysis summary')
    parser.add_argument('--errors-only', action='store_true', help='Show only errors')
    parser.add_argument('--performance', action='store_true', help='Show performance metrics')
    
    args = parser.parse_args()
    
    logs_dir = Path('logs')
    if not logs_dir.exists():
        print("❌ No logs directory found. Make sure the application has been run.")
        return
    
    # Determine which log file to read
    log_files = {
        'app': 'app.log',
        'user': 'user_activity.log',
        'performance': 'performance.log',
        'errors': 'errors.log'
    }
    
    if args.errors_only:
        logs = load_logs(logs_dir / 'errors.log')
        view_errors(logs)
        return
    
    if args.performance:
        logs = load_logs(logs_dir / 'performance.log')
        view_performance(logs)
        return
    
    # Load the specified log type
    log_file = logs_dir / log_files[args.log_type]
    logs = load_logs(log_file)
    
    if args.analyze:
        # Load all logs for comprehensive analysis
        all_logs = []
        for log_file_name in log_files.values():
            file_logs = load_logs(logs_dir / log_file_name)
            all_logs.extend(file_logs)
        analyze_logs(all_logs)
    else:
        view_user_activity(logs, args.user, args.ip)

if __name__ == '__main__':
    main() 