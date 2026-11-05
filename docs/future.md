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
- **Private snapshot store per repo**: a session in a git repo writes its
  checkpoint trees/blobs into *that repo's* object store, as objects no ref
  points at. Nothing in xarness can protect them: the user's own `git gc
  --prune=now` (or `git repack -ad`, or any aggressive prune) discards
  them, and a live session then loses `/diff` and `/undo` until it
  re-baselines. Phase two is to always snapshot into the private store
  (`_managed_store_dir`, today used only for gitless workspaces) with
  `GIT_WORK_TREE` at the repo, so the lifetimes are separate — that needs a
  per-session ref in that store (so a `git gc` there cannot drop a live
  session), an `xarness prune --older-than`, and a migration path for
  sessions whose checkpoints stay behind in the user's repo. Documented as
  current behavior in the README ("Where snapshots live").
- **Auth / Tailscale / TLS**: the API is protected by unix-socket
  permissions only.
