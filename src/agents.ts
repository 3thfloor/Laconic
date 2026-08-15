// 3th Floor AI SDK - Agents
// Author: Justin Bench, 3th Floor AI
//
// ReAct-style tool-use loop. The model signals a tool call by outputting a
// JSON object on its own line:
//
//   {"tool":"name","args":{"param":"value"}}
//
// runAgent feeds that result back as an assistant/user exchange and keeps
// going until the model responds without a tool call or maxTurns is reached.
// The return value is always a plain string.

import type { ChatMessage } from "./engine.js";

export interface Tool {
  name: string;
  description: string;
  fn: (...args: any[]) => any;
}

// Convenience constructor. Lets callers write:
//   tool("add", "Add two numbers", (a, b) => a + b)
// instead of constructing the object literal directly.
export function tool(
  name: string,
  description: string,
  fn: (...args: any[]) => any
): Tool {
  return { name, description, fn };
}

// Shape the model expects when a tool call is signaled.
interface ToolCall {
  tool: string;
  args: Record<string, any>;
}

// Attempt to extract a tool call from the raw model output. The model may
// embed the JSON anywhere in its response, so scan each line. Returns null
// when no valid tool call is found.
function extractToolCall(text: string): ToolCall | null {
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) {
      continue;
    }
    try {
      const parsed = JSON.parse(trimmed);
      if (
        typeof parsed === "object" &&
        parsed !== null &&
        typeof parsed.tool === "string" &&
        typeof parsed.args === "object" &&
        parsed.args !== null
      ) {
        return parsed as ToolCall;
      }
    } catch {
      // Not valid JSON on this line, keep scanning.
    }
  }
  return null;
}

// Build the system prompt that teaches the model the tool-call protocol.
// Appended to the caller-supplied system string when tools are provided.
function buildSystemPrompt(tools: Tool[], callerSystem?: string): string {
  const toolDocs = tools
    .map((t) => `  - ${t.name}: ${t.description}`)
    .join("\n");

  const protocol = [
    "You have access to the following tools:",
    toolDocs,
    "",
    "To call a tool, output exactly one line containing valid JSON in this format:",
    '  {"tool":"<name>","args":{<key>:<value>,...}}',
    "",
    "Output nothing else on that line. After you receive the tool result, continue",
    "reasoning until you can answer without calling another tool.",
    "When you have a final answer, respond normally with no JSON tool-call line.",
  ].join("\n");

  return callerSystem ? `${callerSystem}\n\n${protocol}` : protocol;
}

// Run a ReAct agent loop. Returns the model's final plain-text answer.
//
// engine.chat() is called with a full messages array on every turn so the
// model sees the complete conversation, including tool results. The engine
// creates a fresh context sequence per call, which is the correct behavior
// here because we are managing history ourselves.
export async function runAgent(
  engine: any,
  alias: string,
  userMessage: string,
  options: {
    tools?: Tool[];
    system?: string;
    maxTurns?: number;
  } = {}
): Promise<string> {
  const tools = options.tools ?? [];
  const maxTurns = options.maxTurns ?? 10;
  const toolMap = new Map<string, Tool>(tools.map((t) => [t.name, t]));

  const systemPrompt =
    tools.length > 0
      ? buildSystemPrompt(tools, options.system)
      : options.system;

  // Build the initial message list.
  const messages: ChatMessage[] = [];

  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }

  messages.push({ role: "user", content: userMessage });

  for (let turn = 0; turn < maxTurns; turn++) {
    const response: string = await engine.chat(alias, messages);

    // Record the model turn before deciding what to do next.
    messages.push({ role: "assistant", content: response });

    // No tools registered: always treat the first response as final.
    if (tools.length === 0) {
      return response;
    }

    const call = extractToolCall(response);

    if (!call) {
      // No tool call found. The model is done reasoning.
      return response;
    }

    const matchedTool = toolMap.get(call.tool);

    if (!matchedTool) {
      // Unknown tool name. Tell the model and let it try again.
      const knownNames = Array.from(toolMap.keys()).join(", ");
      const errorNote =
        `Tool "${call.tool}" does not exist. ` +
        `Available tools: [${knownNames || "none"}].`;
      messages.push({ role: "user", content: errorNote });
      continue;
    }

    // Execute the tool. Spread args object values as positional parameters
    // because the fn signature may be positional (add(a, b)) or it may
    // accept a single options object. Pass both: the spread and the object.
    let result: string;
    try {
      const argValues = Object.values(call.args);
      const raw = await Promise.resolve(
        matchedTool.fn(...argValues, call.args)
      );
      result = raw === undefined || raw === null ? "(no result)" : String(raw);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      result = `Tool error: ${message}`;
    }

    // Feed the tool result back as a user turn so the model can continue.
    messages.push({
      role: "user",
      content: `Tool result for "${call.tool}":\n${result}`,
    });
  }

  // maxTurns reached without a clean finish. Return the last assistant reply.
  const lastAssistant = [...messages]
    .reverse()
    .find((m) => m.role === "assistant");

  return lastAssistant?.content ?? "(no response)";
}
