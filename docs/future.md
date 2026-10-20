# Future work

Deliberately left out of the backend/client split; hooks are in place.

- **Web UI**: another pure client of the same API (`clientlib` is the
  reference). It needs a TCP listener on the host (the unix socket only is
  served today) plus Host/Origin checks and an auth token before that.
- **`subagent` tool**: no config section or implementation exists yet;
  this would be a new tool plus host-side child-session support.
- **`notify_command`**: run a user command on turn completion (hook would
  live in `Session._run_turn`'s finally block).
- **Swarm / multi-agent mode**: multiple sessions share a workspace today
  (shared baseline); true multi-agent needs `agent_id` on events and
  per-agent diff tracking.
- **Auth / Tailscale / TLS**: the API is protected by unix-socket
  permissions only.
