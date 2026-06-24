Phase 3 Implementation Tasks
Part 1: Immediate Cleanup
 [DONE] [CHECKMARK] Remove convert_video_to_gif from config.py ACTION_PLAN_SYSTEM_PROMPT
 [DONE] [CHECKMARK] Remove convert_video_to_gif few-shot examples from config.py ACTION_PLAN_EXAMPLES
 [DONE] [CHECKMARK] Remove convert_video_to_gif from main.py action plan fallback prompt
 [DONE] [CHECKMARK] Remove "chroma" from kaia_cli.py restart_service enum
 [DONE] [CHECKMARK] Remove "chroma" from policy_gate.py ALLOWED_SERVICES
 [DONE] [CHECKMARK] Remove "chroma" from host_executor.py ALLOWED_SERVICES
 [DONE] [CHECKMARK] Refresh README.md (ChromaDB standalone reference, test listing)
Part 2: Live InternetDB Lookups
 [NOT DONE JUST STARTED] Add _fetch_internetdb_api() function to threat_intel.py
 Add _cache_internetdb_result() helper
 Add last_updated column to InternetDB schema
 Modify lookup_internetdb() for cache-first, API-fallback logic
 Add INTERNETDB_CACHE_TTL_DAYS constant
Part 3: Dashboard Stage 1
 Read and understand kaiamon.py architecture
 Create kaia_dashboard.py from kaiamon.py
 Repoint PingCollector targets
 Repoint ServiceCollector targets
 Convert LogCollector to AuditLogCollector
 Adapt layout to Kaia mockup
