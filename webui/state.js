/** Transcript reducer — a port of `src/xarness/tui/transcript.py`. A snapshot
 * and the live event stream build the same items, so attach and live turns
 * cannot drift. */
import { splitSettled } from './markdown.js';

export const SUMMARY_PREFIX = '[earlier conversation, summarized]';
export const STATUS = {
  MAKING: 'making_call',
  PARSE_ERROR: 'parsing_error',
  OK: 'call_succeeded',
  FAILED: 'call_failed',
};
export const MAX_LIVE_TOOL_OUTPUT = 65536;
let nextId = 1;
const item = (kind, props) => ({ id: nextId++, kind, ...props });
const userItem = (m) =>
  item('user', { text: messageText(m), images: imageRefs(m), messageId: m.id || null, armed: false });
const thinkingItem = (text, props = {}) =>
  item('thinking', { text, done: false, duration: null, expanded: false, tokens: null, started: null, ...props });
const answerItem = (props = {}) => item('answer', { chunks: [], live: '', finalized: false, ...props });
const toolItem = (callId, name, args = '') =>
  item('tool', { callId, name, args, output: '', status: STATUS.MAKING, expanded: false, usage: null });

export function formatDuration(seconds) {
  const v = Number(seconds) || 0;
  if (v < 10) return `${v.toFixed(1)}s`;
  if (v < 60) return `${Math.floor(v)}s`;
  const mins = Math.floor(v / 60),
    secs = Math.floor(v) % 60;
  if (mins < 60) return `${mins}m${String(secs).padStart(2, '0')}s`;
  return `${Math.floor(mins / 60)}h${String(mins % 60).padStart(2, '0')}m`;
}

export function formatElapsed(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  return total < 60 ? `${total}s` : formatDuration(total);
}

export function formatTokens(count) {
  const v = Number(count) || 0;
  if (v < 1000) return String(v);
  if (v < 1000000) return `${(v / 1000).toFixed(1)}k`;
  return `${(v / 1000000).toFixed(2)}M`;
}

export function formatUsage(usage) {
  if (!usage) return '';
  return `${usage.approximate ? '~' : ''}↑${formatTokens(usage.input_tokens)} ↓${formatTokens(usage.output_tokens)}`;
}

export function imageRefs(message) {
  return (message?.parts || []).filter((p) => p && p.type === 'image' && p.ref).map((p) => p.ref);
}

export function messageText(message) {
  if (!message) return '';
  if (typeof message.content === 'string') return message.content;
  return (message.parts || [])
    .filter((p) => p && p.type === 'text')
    .map((p) => p.text || '')
    .join('');
}

function turnSummaries(messages) {
  const starts = [];
  messages.forEach((m, i) => {
    if (m.role === 'user' && !m.is_summary) starts.push(i);
  });
  const summaries = new Map();
  starts.forEach((start, n) => {
    const turn = messages.slice(start, n + 1 < starts.length ? starts[n + 1] : messages.length);
    const last = [...turn].reverse().find((m) => m.role === 'assistant');
    const stats = last?.timing?.stats;
    if (stats) summaries.set(last, { total: stats.total, busy: stats.busy, idle: stats.idle, usage: stats.usage });
  });
  return summaries;
}
const toolStatusOf = (s) => (s === 'ok' ? STATUS.OK : s === 'parse_error' ? STATUS.PARSE_ERROR : STATUS.FAILED);

export class Transcript {

  constructor() {
    this.items = [];
    this.startRound();
    this.cancelled = false;
  }

  load(snapshot) {
    this.items = [];
    this.startRound();
    const messages = snapshot?.messages || [],
      results = new Map();
    for (const m of messages) if (m.role === 'tool' && m.tool_call_id) results.set(m.tool_call_id, m);
    const summaries = turnSummaries(messages),
      live = snapshot?.status?.state === 'running';
    for (const m of messages) this.addMessage(m, results, summaries.get(m), live);
    this.loadPartial(snapshot?.partial_round);
    return this.items;
  }

  addMessage(message, results, summary, live) {
    if (message.role === 'user') {
      if (message.is_summary)
        this.items.push(item('divider', { text: String(messageText(message)).replace(SUMMARY_PREFIX, '').trim() }));
      else this.items.push(userItem(message));
      return;
    }
    if (message.role !== 'assistant') return;
    const roundUsage = message.usage ? { ...message.usage } : null;
    let roundOwner = null;
    let thoughtOwnsRound = false;
    if (message.reasoning) {
      const thought = thinkingItem(message.reasoning, {
        done: true,
        duration: message.timing?.reasoning_seconds ?? null,
        expanded: false,
        tokens: message.usage?.reasoning_tokens ?? null,
      });
      if (roundUsage) thought.roundUsage = roundUsage;
      roundOwner = thought;
      thoughtOwnsRound = !!roundUsage;
      this.items.push(thought);
    }
    if (message.content || (!message.tool_calls && !message.reasoning)) {
      this.items.push(answerItem({ chunks: [message.content || '(no output)'], finalized: true }));
    }
    for (const call of message.tool_calls || []) {
      const tool = toolItem(call.call_id, call.name, call.arguments_json || '');
      const result = results.get(call.call_id);
      if (result) {
        tool.status = toolStatusOf(result.status);
        tool.output = result.content || '';
        tool.usage = result.usage || null;
      } else if (live) tool.status = STATUS.MAKING;
      else {
        tool.status = STATUS.FAILED;
        tool.output = 'interrupted';
      }
      this.toolBlocks.set(call.call_id, tool);
      this.items.push(tool);
      if (roundUsage && !thoughtOwnsRound) roundOwner = tool;
    }
    // Same rule as the live reducer: the round's spend sits on its thought, or
    // on the last call it asked for when it thought nothing.
    if (roundOwner && !roundOwner.roundUsage) roundOwner.roundUsage = roundUsage;
    if (summary) this.items.push(item('footer', { ...summary }));
  }

  loadPartial(partial) {
    if (!partial) return;
    if (partial.reasoning) {
      this.thinking = thinkingItem(partial.reasoning, { started: performance.now() / 1000 });
      this.items.push(this.thinking);
    }
    if (partial.content) {
      this.assistant = answerItem();
      this.appendDelta(this.assistant, partial.content);
      this.items.push(this.assistant);
    }
    for (const call of partial.tool_calls || []) {
      let tool = this.toolBlocks.get(call.call_id);
      if (!tool) {
        tool = toolItem(call.call_id, call.name || 'tool');
        this.toolBlocks.set(call.call_id, tool);
        this.items.push(tool);
      }
      tool.name = call.name || tool.name;
      tool.args = call.arguments || '';
      tool.output = call.content || '';
      tool.status = call.status === null || call.status === undefined ? STATUS.MAKING : toolStatusOf(call.status);
    }
    if (this.thinking && (partial.content || (partial.tool_calls || []).length)) this.finishThinking(null);
  }

  startRound() {
    this.thinking = null;
    this.assistant = null;
    this.toolBlocks = new Map();
    this.cancelled = false;
    this.reasoningT = null;
  }

  addLine(text, level = 'info') {
    const line = item('line', { text, level });
    this.items.push(line);
    return [line];
  }

  appendDelta(answer, text) {
    answer.live += text;
    const [settled, tail] = splitSettled(answer.live);
    if (settled) {
      answer.chunks.push(settled);
      answer.live = tail;
    }
  }

  finishThinking(duration) {
    const t = this.thinking;
    if (!t || t.done) return null;
    t.done = true;
    t.duration = duration ?? null;
    t.expanded = false;
    return t;
  }

  settleThinking(at) {
    const t = this.thinking;
    if (!t) return null;
    const before = `${t.done}:${t.duration}`;
    this.finishThinking(at !== undefined && at !== null && this.reasoningT !== null ? at - this.reasoningT : null);
    return before === `${t.done}:${t.duration}` ? null : t;
  }

  apply(event) {
    const type = event?.type;
    if (type === 'message_added') {
      const m = event.message;
      if (m.role !== 'user' || m.is_summary) return [];
      const user = userItem(m);
      this.items.push(user);
      return [user];
    }
    if (type === 'processing_started') {
      this.startRound();
      return [];
    }
    if (type === 'turn_ended') {
      if (this.cancelled) return [];
      const changed = this.cancel();
      if (event.stats) {
        const footer = item('footer', {
          total: event.stats.total,
          busy: event.stats.busy,
          idle: event.stats.idle,
          usage: event.stats.usage,
        });
        this.items.push(footer);
        changed.push(footer);
      }
      return changed;
    }
    if (this.cancelled) return [];
    if (type === 'reasoning_delta') {
      if (!this.thinking) {
        this.thinking = thinkingItem('', { started: performance.now() / 1000 });
        this.items.push(this.thinking);
        this.reasoningT = event.t ?? null;
      }
      this.thinking.text += event.text || '';
      return [this.thinking];
    }
    if (type === 'content_delta') {
      const changed = [];
      if (!this.assistant) {
        const settled = this.settleThinking(event.t);
        if (settled) changed.push(settled);
        this.assistant = answerItem();
        this.items.push(this.assistant);
      }
      this.appendDelta(this.assistant, event.text || '');
      changed.push(this.assistant);
      return changed;
    }
    if (type === 'tool_call_started') {
      const settled = this.settleThinking(event.t),
        tool = toolItem(event.call_id, event.name);
      this.toolBlocks.set(event.call_id, tool);
      this.items.push(tool);
      return settled ? [tool, settled] : [tool];
    }
    if (type === 'tool_call_arguments_delta') {
      const tool = this.toolBlocks.get(event.call_id);
      if (!tool) return [];
      tool.args += event.text || '';
      return [tool];
    }
    if (type === 'tool_call_arguments_done') {
      const tool = this.toolBlocks.get(event.call_id);
      if (!tool) return [];
      if (event.name) tool.name = event.name;
      tool.args = event.arguments_json || '';
      return [tool];
    }
    if (type === 'tool_output_delta') {
      const tool = this.toolBlocks.get(event.call_id);
      if (!tool || tool.output.length >= MAX_LIVE_TOOL_OUTPUT) return [];
      tool.output += String(event.text || '').slice(0, MAX_LIVE_TOOL_OUTPUT - tool.output.length);
      return [tool];
    }
    if (type === 'tool_usage_changed' || type === 'tool_finished') {
      const tool = this.toolBlocks.get(event.call_id);
      if (!tool) return [];
      if (type === 'tool_usage_changed') tool.usage = event.usage || null;
      else {
        tool.status = toolStatusOf(event.status);
        tool.output = event.content || '';
      }
      return [tool];
    }
    if (type === 'round_complete') {
      const changed = this.settle(event.t ?? null),
        usage = event.usage || null,
        tokens = usage?.reasoning_tokens || 0,
        t = this.thinking;
      if (t && tokens && tokens !== t.tokens) {
        t.tokens = tokens;
        if (!changed.includes(t)) changed.push(t);
      }
      // The round's own spend belongs to the activity that produced it: the
      // thought it wrote, or failing that the last call it asked for.
      if (usage) {
        const target = t || [...this.toolBlocks.values()].pop() || null;
        if (target && target.roundUsage !== usage) {
          target.roundUsage = usage;
          if (!changed.includes(target)) changed.push(target);
        }
      }
      // A round is one model response, so the thought and the answer it streamed
      // end here: the next round opens its own items, exactly as a snapshot of
      // the same turn rebuilds them one assistant message at a time.
      this.thinking = null;
      this.assistant = null;
      this.reasoningT = null;
      return changed;
    }
    if (type === 'stream_error') return this.settle(null);
    return [];
  }

  settle(at) {
    const changed = [],
      settled = this.settleThinking(at);
    if (settled) changed.push(settled);
    if (this.assistant) {
      if (!this.assistant.finalized) {
        this.assistant.finalized = true;
        const tail = this.assistant.live;
        if (tail.trim()) {
          this.assistant.chunks.push(tail);
          this.assistant.live = '';
        } else if (!this.assistant.chunks.length) this.assistant.chunks.push('(no output)');
      }
      changed.push(this.assistant);
    }
    return changed;
  }

  cancel() {
    this.cancelled = true;
    const changed = this.settle(null);
    for (const tool of this.toolBlocks.values()) {
      if (tool.status === STATUS.MAKING) {
        tool.status = STATUS.FAILED;
        tool.output = tool.output || 'interrupted';
        changed.push(tool);
      }
    }
    return changed;
  }
}
