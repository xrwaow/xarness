/** Slash commands — the registry of `COMMANDS` in `src/xarness/tui/app.py`.
 * Enter runs the highlighted command; Tab completes an argument-taking one. */
const cmd = (name, description, run, usage = '', argHint = '') => ({
  name,
  description,
  run,
  ...(usage ? { usage, argHint } : {}),
});
export const COMMANDS = [
  cmd('tools', 'browse tools and manage MCP servers', (a) => a.openTools()),
  cmd('model', 'choose what model and reasoning effort to use', (a) => a.pickModel()),
  cmd('sessions', 'resume a previous session', (a) => a.openSessions()),
  cmd('mode', 'switch between plan (read-only) and write mode', (a) => a.toggleMode()),
  cmd('sandbox', "choose this session's sandbox profile (idle only)", (a) => a.pickSandbox()),
  cmd(
    'filter',
    'add/remove/list paths excluded from the sandbox mounts',
    (a, x) => a.filterCommand(x),
    '[add|remove] <path>',
    'a host path',
  ),
  cmd('refs', "add/remove this workspace's external reference paths", (a) => a.openRefs()),
  cmd('theme', 'choose a color theme', (a, x) => a.handleTheme(x), '[name]', 'a theme name'),
  cmd('new', 'leave this session and start a fresh chat', (a) => a.newChat()),
  cmd('delete', 'delete the current session and start a new one (your files are left exactly as they are)', (a) =>
    a.deleteCurrent(),
  ),
  cmd(
    'diff',
    'show pending changes (optionally: /diff <path>)',
    (a, x) => a.showDiffCommand(x),
    '[path]',
    'a file path',
  ),
  cmd('accept', 'lock in the changes made so far (they stop showing in /diff and can no longer be undone)', (a) =>
    a.accept(),
  ),
  cmd('reject', 'discard all changes made since the last /accept', (a) => a.reject()),
  cmd('undo', 'drop the last turn and revert its file edits', (a) => a.undo()),
  cmd('retry', "revert the last turn's file edits and resend your message", (a) => a.retry()),
  cmd('help', 'keyboard shortcuts and what this UI can do', (a) => a.openHelp()),
];
export const findCommand = (name) => COMMANDS.find((c) => c.name === name) || null;
export const commandUsage = (c) => `/${c.name}${c.usage ? ` ${c.usage}` : ''}`;
