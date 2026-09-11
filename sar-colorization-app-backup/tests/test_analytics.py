"""Truthfulness, privacy, retention, and API tests for research analytics."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend import app
from satquery_agent.analytics import ANALYTICS_STORE, MAX_HISTORY, build_analytics, record_agent_response
from satquery_agent.models import (
    AgentResponse,
    AnalyticsExecution,
    AnalyticsTraceStep,
    CacheMetadata,
    Confidence,
    ConfidenceLevel,
    ExecutionStep,
    ExecutionSummary,
    InputMode,
    Modality,
    ResponseStatus,
    TaskType,
    ToolStatus,
    ValidationStatus,
)
from satquery_agent.reporting import MISSION_STORE


def response(request_id: str = "safe-request", *, cached: bool = False) -> AgentResponse:
    return AgentResponse(
        request_id=request_id,
        task=TaskType.VQA,
        answer="private answer must never be retained",
        confidence=Confidence(level=ConfidenceLevel.LOW, score=None, reason="not calibrated"),
        evidence=[],
        execution=ExecutionSummary(
            input_mode=InputMode.SINGLE,
            selected_tools=["input_validator", "rs_vqa"],
            steps=[ExecutionStep(tool="input_validator", status=ToolStatus.SUCCESS, duration_ms=2, parameters={"secret": "never"})],
            duration_ms=12,
            permitted_parameters={"query": "private query"},
            validation=ValidationStatus(valid=True, errors=[]),
            selection_reason="test",
        ),
        warnings=["measured warning"],
        status=ResponseStatus.SUCCESS,
        cache=CacheMetadata(
            cached=cached,
            original_generation_timestamp="2026-01-01T00:00:00Z",
            retrieval_timestamp="2026-01-01T00:00:01Z",
            tool_version="test",
            cache_key_prefix="abcdef123456",
        ),
    )


def execution(index: int, task: str = "vqa") -> AnalyticsExecution:
    return AnalyticsExecution(
        request_id=f"request-{index}",
        started_at=f"2026-01-01T00:00:{index % 60:02d}Z",
        completed_at=f"2026-01-01T00:01:{index % 60:02d}Z",
        task=task,
        input_mode="single",
        primary_modality="optical",
        status="success",
        selected_tools=["rs_vqa"],
        duration_ms=index,
        warning_count=0,
        output_count=0,
        cache_status="fresh",
        trace=[AnalyticsTraceStep(tool="rs_vqa", status="success", duration_ms=index)],
    )


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        ANALYTICS_STORE.clear()
        MISSION_STORE.clear()

    def test_endpoint_is_available(self):
        result = self.client.get("/api/agent/analytics")
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn("scientific_transparency", result.json())

    def test_empty_history_is_explicit(self):
        body = build_analytics()
        self.assertEqual(body.recent_executions, [])
        self.assertEqual(body.last_execution_trace, [])
        self.assertEqual(body.summary.total_executions_current_process, 0)

    def test_registry_and_compliance_counts_are_derived(self):
        body = build_analytics()
        self.assertEqual(body.summary.registered_tools, len(body.tools))
        self.assertEqual(body.summary.mandatory_satisfied, 16)
        self.assertEqual(body.summary.mandatory_total, 16)

    def test_unavailable_memory_is_null_not_zero(self):
        self.assertIsNone(build_analytics().platform.process_memory_mb)

    def test_dataset_counts_are_not_fabricated(self):
        datasets = {item.name: item for item in build_analytics().datasets}
        self.assertEqual(datasets["BigEarthNet.txt / BigEarthNet v2 Lithuania Summer"].sample_count, 4008)
        self.assertTrue(all(
            item.sample_count is None
            for name, item in datasets.items()
            if name != "BigEarthNet.txt / BigEarthNet v2 Lithuania Summer"
        ))

    def test_sar_training_dataset_is_not_claimed(self):
        item = next(item for item in build_analytics().datasets if item.name == "SAR-to-optical training corpus")
        self.assertEqual(item.usage_status, "Undeclared")
        self.assertIn("SEN12MS-CR is not claimed", item.note)

    def test_rsvqa_is_not_presented_as_active(self):
        item = next(item for item in build_analytics().datasets if item.name == "RSVQA")
        self.assertEqual(item.usage_status, "Evaluation Candidate / Not Used")
        self.assertEqual(item.used_by, [])

    def test_grounding_provenance_is_not_a_project_dataset_claim(self):
        item = next(item for item in build_analytics().datasets if item.name.startswith("Grounding DINO"))
        self.assertEqual(item.usage_status, "Model Provenance")
        self.assertIn("not a GeoVision project dataset", item.note)

    def test_recorded_history_omits_query_answer_and_parameters(self):
        record_agent_response(
            response(),
            started_at="2026-01-01T00:00:00Z",
            primary_modality=Modality.OPTICAL.value,
            secondary_modality=None,
        )
        serialized = build_analytics().model_dump_json()
        self.assertNotIn("private query", serialized)
        self.assertNotIn("private answer", serialized)
        self.assertNotIn('"secret"', serialized)

    def test_safe_trace_parameters_are_whitelisted(self):
        item = response()
        item.execution.steps[0].parameters = {"device": "cpu", "secret": "never", "target_phrase": "private query"}
        record_agent_response(item, started_at="2026-01-01T00:00:00Z", primary_modality="optical", secondary_modality=None)
        parameters = build_analytics().recent_executions[0].trace[0].parameters
        self.assertEqual(parameters, {"device": "cpu"})

    def test_routing_confidence_and_warnings_are_observable(self):
        record_agent_response(response(), started_at="2026-01-01T00:00:00Z", primary_modality="optical", secondary_modality=None)
        item = build_analytics().recent_executions[0]
        self.assertEqual(item.selection_reason, "test")
        self.assertEqual(item.confidence_level, "low")
        self.assertEqual(item.warnings, ["measured warning"])

    def test_history_is_newest_first(self):
        ANALYTICS_STORE.record(execution(1))
        ANALYTICS_STORE.record(execution(2))
        identifiers = [item.request_id for item in build_analytics().recent_executions]
        self.assertEqual(identifiers[:2], ["request-2", "request-1"])

    def test_history_is_bounded_to_fifty(self):
        for index in range(MAX_HISTORY + 7):
            ANALYTICS_STORE.record(execution(index))
        history = build_analytics().recent_executions
        self.assertEqual(len(history), MAX_HISTORY)
        self.assertEqual(history[0].request_id, f"request-{MAX_HISTORY + 6}")

    def test_total_counter_can_exceed_retained_history(self):
        for index in range(MAX_HISTORY + 2):
            ANALYTICS_STORE.record(execution(index))
        body = build_analytics()
        self.assertEqual(body.summary.total_executions_current_process, MAX_HISTORY + 2)
        self.assertEqual(len(body.recent_executions), MAX_HISTORY)

    def test_last_trace_comes_from_newest_execution(self):
        ANALYTICS_STORE.record(execution(4))
        ANALYTICS_STORE.record(execution(8))
        self.assertEqual(build_analytics().last_execution_trace[0].duration_ms, 8)

    def test_workflow_averages_use_real_runs_only(self):
        ANALYTICS_STORE.record(execution(10))
        ANALYTICS_STORE.record(execution(20))
        metric = next(item for item in build_analytics().workflow_metrics if item.task == "vqa")
        self.assertEqual(metric.executions, 2)
        self.assertEqual(metric.average_runtime_ms, 15.0)

    def test_unexecuted_workflow_runtime_is_null(self):
        metric = next(item for item in build_analytics().workflow_metrics if item.task == "grounding")
        self.assertIsNone(metric.average_runtime_ms)
        self.assertIsNone(metric.last_runtime_ms)

    def test_report_counter_marks_matching_history(self):
        ANALYTICS_STORE.record(execution(1))
        ANALYTICS_STORE.record_report("request-1", ["pdf", "json", "zip"])
        body = build_analytics()
        self.assertEqual(body.reports.requests_generated_current_process, 1)
        self.assertEqual(body.reports.artifacts_generated_current_process, 3)
        self.assertTrue(body.recent_executions[0].report_generated)
        self.assertEqual(body.reports.formats, {"pdf": 1, "json": 1, "zip": 1})

    def test_report_counter_does_not_mark_unrelated_history(self):
        ANALYTICS_STORE.record(execution(1))
        ANALYTICS_STORE.record_report("different-request", ["csv"])
        self.assertFalse(build_analytics().recent_executions[0].report_generated)

    def test_cache_miss_counter_counts_real_lookups(self):
        self.assertIsNone(MISSION_STORE.get_by_cache("absent"))
        self.assertEqual(build_analytics().cache.misses, 1)

    def test_cache_hit_counter_counts_real_lookups(self):
        item = response()
        MISSION_STORE.put(item, "known-cache-key")
        self.assertIsNotNone(MISSION_STORE.get_by_cache("known-cache-key"))
        body = build_analytics()
        self.assertEqual(body.cache.hits, 1)
        self.assertEqual(body.cache.hit_rate_percent, 100.0)

    def test_cache_hit_rate_is_null_without_lookups(self):
        self.assertIsNone(build_analytics().cache.hit_rate_percent)

    def test_cache_clear_resets_counters(self):
        MISSION_STORE.get_by_cache("absent")
        MISSION_STORE.clear()
        self.assertEqual(build_analytics().cache.misses, 0)

    def test_capability_gaps_remain_visible(self):
        capabilities = {item.name: item.status for item in build_analytics().capabilities}
        self.assertEqual(capabilities["Grounding masks"], "Not Implemented")
        self.assertEqual(capabilities["Automatic image registration"], "Not Implemented")
        self.assertEqual(capabilities["SAR captioning"], "Unsupported")

    def test_model_lifecycle_is_separate_from_implementation(self):
        captioner = next(item for item in build_analytics().tools if item.id == "rs_captioner")
        self.assertEqual(captioner.implementation_status, "available")
        self.assertIn(captioner.lifecycle_status, {"unloaded", "ready", "failed"})

    def test_direct_change_tool_alias_populates_registry_runtime(self):
        item = execution(9, task="change_analysis").model_copy(update={"selected_tools": ["input_validator", "deterministic_change_analysis"]})
        ANALYTICS_STORE.record(item)
        change_tool = next(tool for tool in build_analytics().tools if tool.id == "bitemporal_change_analyzer")
        self.assertEqual(change_tool.last_runtime_ms, 9)

    def test_runtime_versions_are_detected(self):
        versions = build_analytics().platform.runtime_versions
        self.assertIsNotNone(versions["fastapi"])
        self.assertIsNotNone(versions["rasterio"])

    def test_endpoint_response_does_not_expose_local_paths(self):
        body = self.client.get("/api/agent/analytics").text
        self.assertNotIn("/Users/", body)
        self.assertNotIn("SATQUERY_MODEL_CACHE", body)


if __name__ == "__main__":
    unittest.main()
