# TMon CLI Design Document

1. Overview  
TMon (“Telemetry Monitor”) is a fast, lightweight Python CLI tool built on top of the existing Jetson Power GUI core libraries. It provides deterministic, high-frequency polling of on-device sensors and logs or streams results without any GUI overhead.

2. Goals & Non-Goals  
• Deterministic timing: achieve minimal jitter at millisecond resolution  
• Re-use DataCenter & DataLogger for sensor I/O, conversion and CSV formatting  
• Minimal dependencies: Click for CLI, stdlib threading/time, pylibjetsonpower under the hood  
• Modular design: clear separation between CLI parsing, polling core, and logging/streaming  
• Extensible: easy to add new commands (e.g. “alert”, “profile”)  
• Not a daemon manager: process supervision left to systemd/docker  

3. High-Level Architecture  
```
   ┌──────────────┐  
   │   CLI Layer  │    ← parses commands & options via Click  
   └──────┬───────┘  
          │  
   ┌──────▼───────┐  
   │  Controller  │    ← orchestrates polling loop & commands (start/stop/status)  
   └──────┬───────┘  
          │  
   ┌──────▼───────┐  
   │ Polling Core │    ← wraps DataCenter.update_data + sleep strategy  
   └──────┬───────┘  
          │  
   ┌──────▼───────┐       ┌─────────────┐  
   │ DataCenter   │──────▶│ DataLogger  │  
   │ (Subject)    │       │ (Facade for CSV I/O)  
   └──────────────┘       └─────────────┘  
```

4. Components & Design Patterns  
• CLI Layer (Click) with a `tmon` group and commands: start, stop, status, list‐channels, reset  
• Controller using the Command pattern for each CLI command  
• Polling Core using the Strategy pattern to implement a timestamp‐anchored sleep→poll loop  
• DataCenter & DataLogger reused as Subject/Facade for sensor I/O and CSV formatting  
• Configuration via dependency injection and optional `~/.tmonrc` YAML  

5. Command Reference  
• `tmon start` ‑-period INT ‑-duration INT ‑-output PATH [--background]  
• `tmon stop`  
• `tmon status`  
• `tmon list-channels`  
• `tmon reset`  

6. Error Handling & Logging  
• Exit codes: 0=success, 1=usage, 2=runtime error  
• Structured stderr logging via Python’s `logging`  
• Graceful cleanup on SIGINT/SIGTERM with pidfile removal  

7. Performance & Determinism  
• Single-threaded polling loop with monotonic timer  
• Keep file handle open, batch I/O, optional stdout streaming  

8. Testing Strategy  
• Unit tests for Controller, CLI parsing, timing strategies  
• Integration tests with a stub DataCenter  
• CI: lint (flake8), type‐check (mypy), coverage threshold  

9. Deployment & Distribution  
• `pip install jetsonpowergui[tmon]` entry point `tmon=jetsonpowergui.cli:main`  
• RPM/deb packages for NVIDIA Linux  

10. Next Steps  
1. Stub out `cli.py` with Click commands  
2. Extract polling core into `utils/tmon/core.py`  
3. Wire DataCenter/DataLogger injection  
4. Implement pidfile‐based stop/status  
5. Write tests and tune performance  

— End of Design Document —
