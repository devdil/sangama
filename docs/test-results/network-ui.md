# Network UI acceptance — 27 September 2026

- All 23 isolated Docker mesh checks passed, including protected UI allocation, live admitted/relay telemetry and real Qwen generation through reversed candidate aliases. UI token IDs matched the independent baseline. Existing delay, disconnection, recovery, quota, revocation, authority-expiry and actual OpenCode checks also passed.
- Ten local PostgreSQL/portal checks passed: operator authentication, Origin protection, role validation, invitation hashing, membership listing, credential non-reflection, revocation confirmation/persistence, onboarding and rate limiting. Disposable rows were removed after the test.
- Root library tests (13) and portal tests (3) passed. Clippy passed for root Metal and portal all-target builds. JavaScript syntax, Python syntax and documentation links were checked.
- Browser inspection verified the classic portal form, onboarding page and local controls; peer actions were disabled without configured bridges. The optimized Metal executable was rebuilt.

Detailed reports: [mesh + UI](ui-mesh-simulation.json), [portal controls](portal-admin.json).

Scope: local simulation, not two physical home networks. UI allocation reports an operation phase, not per-file percentages. Connection telemetry reports observed connections, not the chosen path of every packet. Worker enrollment still signs its challenge locally; the portal never receives private peer keys. Browser administration uses a single separate operator credential, not multi-user audited accounts or MFA.
