#!/usr/bin/env python3
"""
Safer Dash runner script.

This script runs the Dash app with better resource management and cleanup.
"""

import os
import sys
import atexit
import time
import signal
import subprocess
import psutil

def find_dash_process():
    """Find and return any running Dash processes."""
    dash_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if this is a Python process running dash_app.py
            if proc.info['name'].lower() in ('python', 'python3', 'pythonw'):
                cmdline = ' '.join(proc.info['cmdline']).lower() if proc.info['cmdline'] else ''
                if 'dash_app.py' in cmdline:
                    dash_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return dash_processes

def kill_existing_dash():
    """Find and kill any existing Dash processes."""
    dash_processes = find_dash_process()
    if dash_processes:
        print(f"Found {len(dash_processes)} running Dash processes. Terminating...")
        for proc in dash_processes:
            try:
                print(f"Terminating process {proc.pid}")
                proc.terminate()
                # Give it a moment to terminate gracefully
                proc.wait(timeout=3)
            except Exception as e:
                print(f"Error terminating process {proc.pid}: {e}")
                # Force kill if terminate fails
                try:
                    proc.kill()
                except:
                    pass
        # Double-check all processes were killed
        time.sleep(1)
        remaining = find_dash_process()
        if remaining:
            print(f"Warning: {len(remaining)} Dash processes still running.")
        else:
            print("All Dash processes terminated successfully.")

def cleanup():
    """Clean up resources when the script exits."""
    print("\nCleaning up resources...")
    if dash_process:
        try:
            # Try to terminate gracefully first
            dash_process.terminate()
            # Wait briefly for it to terminate
            try:
                dash_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If it doesn't terminate, force kill
                print("Dash app not responding. Force killing...")
                dash_process.kill()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    # Kill any other remaining Dash processes
    kill_existing_dash()
    print("Cleanup complete.")

if __name__ == "__main__":
    # Register the cleanup function to run on exit
    atexit.register(cleanup)
    
    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: sys.exit(0))
    
    # Kill any existing Dash processes first
    kill_existing_dash()
    
    # Get the path to dash_app.py
    dash_app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dash_app.py")
    
    print(f"Starting Dash app from {dash_app_path}")
    print("Navigate to http://127.0.0.1:8050/ in your web browser.")
    print("Press Ctrl+C to stop the server and clean up.")
    
    # Start the Dash app as a subprocess
    dash_process = subprocess.Popen(
        [sys.executable, dash_app_path],
        # Use non-debug mode for better performance and less memory usage
        env={**os.environ, "DASH_DEBUG": "false"}
    )
    
    try:
        # Wait for the process to complete
        dash_process.wait()
    except KeyboardInterrupt:
        print("\nReceived interrupt, shutting down...")
        # cleanup will be called by atexit 