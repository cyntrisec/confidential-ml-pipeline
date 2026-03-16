# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in confidential-ml-pipeline, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please use one of these channels:

1. **GitHub Security Advisories** (preferred): [Report a vulnerability](https://github.com/cyntrisec/confidential-ml-pipeline/security/advisories/new)
2. **Email**: security@cyntrisec.com

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 7 days
- **Fix**: Depends on severity; critical issues prioritized

## Scope

- **Pipeline orchestration** (stage-to-stage relay, measurement enforcement, activation forwarding)
- **Tensor metadata handling** (zeroization, memory safety)
- **Protocol security** (envelope framing, size guards, version enforcement)
- **Key material handling** (session keys, shard manifest secrets)

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| < 0.4   | No        |