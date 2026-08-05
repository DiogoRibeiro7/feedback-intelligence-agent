# Security Policy

## Supported versions

Security fixes are applied to the current `main` branch and the latest GitHub
release.

## Reporting a vulnerability

Please do not open a public issue for security vulnerabilities. Use GitHub
Security Advisories:

https://github.com/DiogoRibeiro7/feedback-intelligence-agent/security/advisories/new

Include:

- Affected component or workflow.
- Reproduction steps or proof of concept.
- Expected impact.
- Any known mitigations.

## Project security model

The default demo runs locally without API keys or managed services. Optional LLM
and vector database integrations must be configured explicitly through
environment variables. Do not include real customer data, secrets, API keys, or
private deployment configuration in issues, pull requests, examples, or tests.
