## Overview
This plugin provides ChatGPT Codex models for Dify through the ChatGPT subscription-backed Codex interface.
It is intentionally incompatible with the standard OpenAI API: it only supports the Codex Responses API, Codex-compatible authentication, and the same model set allowed by the `opencode` Codex plugin.

Supported models:

- `gpt-5.1-codex`
- `gpt-5.1-codex-max`
- `gpt-5.1-codex-mini`
- `gpt-5.2`
- `gpt-5.2-codex`
- `gpt-5.3-codex`
- `gpt-5.4`
- `gpt-5.4-mini`

## Configure
After installing the plugin, configure the provider with:

- a ChatGPT Codex access token
- a ChatGPT Codex refresh token
- an optional ChatGPT account id
- an optional custom Codex API base

The plugin defaults to `https://chatgpt.com/backend-api/codex` and validates credentials with the `gpt-5.3-codex` model unless you override `validate_model`.
