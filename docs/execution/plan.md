<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Execution record — 12 November 2026

1. Audit current remote heads, owner reviews and baseline tests. Preserve C10/C11 history; propose #5 as canonical without closing #3/#4.
2. Extend the existing C9 descriptor mechanism through a proposed component-profile adjunct in interfaces, keeping v1 bytes unchanged. Model names/limits are evidence; command capture and policy authorization remain owner decisions.
3. Implement shared pre-export checks and mutation proof, then observational bag recording and explicit offline alignment/native conversion. Block unresolved profiles for actual recordings.
4. Exercise actual pinned LeRobot writer/loader in an isolated environment; run C10/C11 synthetic software smoke separately from real-data readiness.
5. Add checkpoint-bound inference timing and device telemetry; exercise locally and provide ten-minute A/B/C commands for the hardware owners.
6. Run regressions, package/lint checks and current-head CI. Publish focused drafts and concise updates with remaining evidence gates.

Baseline: embodied-AI 59bb4bf: 166 tests + 17 subtests PASS. Interfaces current main 14912e1: full verify BLOCKED at absent ROS Jazzy prerequisites; portable schema suite initially 12 pass / 3 errors (catkin_pkg missing), then 12 pass / 3 errors (rosidl_adapter absent). No changes to the verifier or gate.

Read-only sources: upperbody 2ca739202f5c51114de1520902e811817d9b2a62; platform 713574fd2a5a9e38cde6b07c5b96ac4f380b3402; manipulation be76d30632f01aab2badddf4e33de829c113b1b7; governance main 87c8486. `/opt/ros` absent; Docker daemon unavailable. No configured Jetson SSH alias found; no physical access established.

Dependency/approval gates: upperbody#35 software-lead adoption; #36 reviewed stowed pose and CTL-017 telemetry; #37 camera/topic bindings; platform#116 requested/accepted/applied distinction and pending policy input; interfaces#30 I8 remains open. LeRobot v0.6.1 is an inspected development pin, not a project-approved release. It requires Python >=3.12 and Torch <2.12, unlike the existing local Torch 2.15 environment; native-loader verification uses a separate environment.
