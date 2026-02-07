# Contributing to confidential-ml-pipeline

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork and create a feature branch:
   ```bash
   git checkout -b my-feature
   ```
3. Make your changes
4. Run the test suite:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```
5. Commit and push to your fork
6. Open a pull request

## Development Requirements

- Rust stable (edition 2021)
- `tokio` runtime (tests use `#[tokio::test]`)

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy -- -D warnings` and fix all warnings
- Write tests for new functionality
- Keep `unsafe` usage to zero in this crate

## Testing

```bash
# Unit + integration tests
cargo test

# TCP integration tests only
cargo test --test tcp_pipeline

# With debug logging
RUST_LOG=debug cargo test -- --nocapture
```

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for bug fixes and new features
- Update the README if adding user-facing functionality
- Ensure CI passes (tests, clippy, fmt)

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).
