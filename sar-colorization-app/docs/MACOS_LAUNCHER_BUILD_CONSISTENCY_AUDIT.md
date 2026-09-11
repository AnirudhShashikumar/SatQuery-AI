# macOS launcher and frontend build consistency audit

Date: 2026-09-06

## Observed processes before repair

- Port 3000 was owned by PID 8513, `next-server (v15.5.23)`, whose parent was PID 8499, `npm run start --hostname 127.0.0.1 --port 3000`. Both used the repository `frontend/` directory as their working directory and had started at 00:11:11.
- Port 3011 had no remaining listener at the start of this repair. It had held the isolated `npm run dev -- --port 3011` process started explicitly for the immediately preceding grounding/bi-temporal browser verification; that process was stopped after the verification.
- The old launcher state named port 3000 but contained stale PID files (`frontend.pid` 7613 while the observed listener was 8513) and only recorded that a URL responded. It had no build identity.

## Root cause

`start_satquery.sh` treated any successful HTTP response on port 3000 as the current SatQuery frontend. It did not verify the listener command, working directory, launcher state, Next production build, or source/build identity. Rebuilding `.next` therefore did not cause the installed app to replace the already-running server, and repeated app launches continued to open the old process.

## Repair

- The installer always performs a fresh production `next build` and generates a timestamp + git SHA build identity.
- The identity, git SHA, and centralized API URL are embedded in the build and served from `/api/build-info` with no-store caching.
- The launcher verifies the endpoint plus exact process command and working directory before reuse.
- A verified stale SatQuery frontend on port 3000 is stopped by exact PID. An unrelated listener is never stopped; the launcher selects the first free port from 3001 through 3010 and opens that exact port.
- Alternate-port reuse requires matching saved state and current build identity, preventing an unrecorded stale server on 3011 from being mistaken for the installed instance.
- State now records backend/frontend PIDs and ports, frontend build identity, git SHA, project root, launch timestamp, URL, and ownership flags.
- An atomic launch lock serializes duplicate Finder launches.
- The stop script acts only on PIDs recorded as launcher-owned and re-verifies command and working directory immediately before termination.
- No service worker or PWA runtime cache was found. Normal Next.js fingerprinted assets remain in use; browser cache clearing is not part of the fix.

## End-to-end result

The final installer run produced build `2c66d7aace92-20260906T135147Z-dirty`, refreshed `~/Applications/SatQuery AI.app`, and launched it from Finder-equivalent macOS application invocation. The production frontend reported the same identity and `http://127.0.0.1:8010` API target. Grounding DINO and Grounding Specialist v1.1 appeared ready from local MPS-backed bundles, and the current bi-temporal hybrid interface and built-up query controls were visible.

A second launch reused the same backend and frontend PIDs. In a real port-collision drill, a controlled unrelated Python server remained alive on 3000 while SatQuery started on 3001, wrote `frontend_port: 3001`, and opened Safari at 3001. After launcher-owned processes and the controlled test server were stopped, a final launch restored the canonical ports: backend 8010 and frontend 3000.

A final live rebuild test kept the earlier production server running with build ID `2c66d7aace92-20260906T134459Z-dirty`, rebuilt and reinstalled the app, and then launched again. The launcher observed the build-ID mismatch, verified that PID 18735 was a SatQuery `next-server` in the expected frontend directory, terminated only that PID, and started the new build on PID 19216. The backend remained on PID 18641 and was safely reused.

## Scope and integrity

Only the macOS installer/launcher lifecycle, frontend build identity endpoint, local identity ignores, tests, and launcher documentation changed. Backend APIs, model logic, checkpoints, Grounding behavior, bi-temporal behavior, scientific safeguards, user data, and report generation were not modified for this repair.
