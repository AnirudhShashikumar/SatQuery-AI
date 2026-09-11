# TTP production smoke

- Status: **failed**
- Error: `RuntimeError: TTP did not execute as primary; engine={'mode': 'deterministic_fallback', 'primary_tool': 'deterministic_change_analyzer', 'supporting_tool': None, 'fallback_used': True, 'fallback_reason': 'unavailable'}`

- Runtime before failure: `1392.5 ms`
- TTP health: `{"status": "unavailable", "lifecycle": "unavailable", "device": null, "checkpoint": "epoch_260.pth", "loaded": false, "load_count": 0, "reuse_count": 0, "inference_count": 0, "errors": ["TTP is unavailable; deterministic fallback was used."]}`

The runner does not count deterministic fallback as a learned-engine pass.
