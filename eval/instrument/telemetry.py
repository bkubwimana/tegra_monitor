import subprocess
import os
import signal
import time
from typing import Optional

class TelemetryHandler:
    def __init__(self, output_dir: str, config_suffix: str = ""):
        self.output_dir = output_dir
        self.config_suffix = config_suffix
        self.log_file = os.path.join(output_dir, f"tegrastats_{config_suffix}.log")
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
    def start_logging(self):
        """Start tegrastats logging"""
        try:
            # Stop any existing tegrastats processes first
            subprocess.run(['tegrastats', '--stop'], 
                         capture_output=True, check=False)
            time.sleep(1)
            
            # Start new logging - use the same format as your working telemetry.sh
            cmd = ['tegrastats', '--start', '--interval', '1000', 
                   '--logfile', self.log_file]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: tegrastats start failed: {result.stderr}")
            
            time.sleep(2)  # Give it time to start
            print(f"Started tegrastats logging to: {self.log_file}")
            
        except Exception as e:
            print(f"Error starting tegrastats: {e}")
            
    def stop_logging(self):
        """Stop tegrastats logging"""
        try:
            result = subprocess.run(['tegrastats', '--stop'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: tegrastats stop failed: {result.stderr}")
                
            print(f"Stopped tegrastats logging")
            time.sleep(2)  # Give it time to finish writing
            
            # Verify log file was created
            if os.path.exists(self.log_file):
                file_size = os.path.getsize(self.log_file)
                print(f"Telemetry log created: {self.log_file} ({file_size} bytes)")
            else:
                print(f"Warning: Telemetry log file not found: {self.log_file}")
                
        except Exception as e:
            print(f"Error stopping tegrastats: {e}")
            
    def get_log_file(self) -> str:
        """Get the path to the log file"""
        return self.log_file
        
    def __enter__(self):
        self.start_logging()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_logging()
