// 3th Floor AI SDK - Engine
// Author: Justin Bench, 3th Floor AI
//
// The Engine holds models in memory and answers questions with plain strings.
// No daemons, no ports, no subprocesses. Load a model, ask it things.

import {
  getLlama,
  LlamaChatSession,
  type ChatHistoryItem,
  type Llama,
  type LlamaModel,
  type LlamaContext,
} from "node-llama-cpp";
import { Session } from "./session.js";
import { ModelManager } from "./models.js";

export interface ChatMessage {
  role: string;
  content: string;
}

export interface ChatOptions {
  system?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface LoadOptions {
  ctx?: number;
  gpuLayers?: number;
  useMmap?: boolean;
}

export interface EngineOptions {
  modelsDir?: string;
}

interface LoadedModel {
  model: LlamaModel;
  context: LlamaContext;
  meta: { alias: string; path: string; ctx: number; loadedAt: number };
}

const DEFAULT_CTX = 4096;
const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_MAX_TOKENS = 2048;

// Sentinel object that marks the end of a token stream.
const STREAM_DONE = Symbol("stream-done");

export class Engine {
  private _llama: Llama | null = null;
  private _loaded: Map<string, LoadedModel> = new Map();
  public models: ModelManager;

  constructor(options: EngineOptions = {}) {
    this.models = new ModelManager(this, options.modelsDir);
  }

  private async getLlama(): Promise<Llama> {
    if (!this._llama) {
      this._llama = await getLlama();
    }
    return this._llama;
  }

  // Load a GGUF model into memory under an alias. Returns the engine so
  // calls can chain: await engine.load("a", pathA) then engine.chat("a", ...).
  async load(alias: string, path: string, options: LoadOptions = {}): Promise<Engine> {
    if (this._loaded.has(alias)) {
      throw new Error(
        `Model alias "${alias}" is already loaded. Call unload("${alias}") first or pick a different alias.`
      );
    }

    const ctx = options.ctx ?? DEFAULT_CTX;
    const llama = await this.getLlama();
    const model = await llama.loadModel({
      modelPath: path,
      ...(options.gpuLayers !== undefined && { gpuLayers: options.gpuLayers }),
      ...(options.useMmap !== undefined && { useMmap: options.useMmap }),
    });
    const context = await model.createContext({ contextSize: ctx });

    this._loaded.set(alias, {
      model,
      context,
      meta: {
        alias,
        path,
        ctx,
        loadedAt: Date.now(),
      },
    });

    return this;
  }

  // Unload a model and free its memory.
  async unload(alias: string): Promise<Engine> {
    const entry = this._loaded.get(alias);
    if (!entry) {
      throw new Error(
        `Model alias "${alias}" is not loaded. Loaded models: [${this.loaded().join(", ")}]`
      );
    }

    this._loaded.delete(alias);
    await entry.context.dispose();
    await entry.model.dispose();

    return this;
  }

  // Ask a loaded model a question. Returns a plain string. That is the
  // whole point: no envelope objects, no digging through response shapes.
  async chat(
    alias: string,
    message: string | ChatMessage[],
    options: ChatOptions = {}
  ): Promise<string> {
    const entry = this.requireLoaded(alias);
    const { systemPrompt, userText, priorHistory } = this.resolveMessage(message, options.system);

    const sequence = entry.context.getSequence();
    try {
      const session = new LlamaChatSession({
        contextSequence: sequence,
        systemPrompt,
      });

      if (priorHistory.length > 0) {
        session.setChatHistory(priorHistory);
      }

      const response = await session.prompt(userText, {
        temperature: options.temperature ?? DEFAULT_TEMPERATURE,
        maxTokens: options.maxTokens ?? DEFAULT_MAX_TOKENS,
      });

      return response;
    } finally {
      sequence.dispose();
    }
  }

  // Stream tokens as they generate. Yields plain string chunks.
  //
  // node-llama-cpp delivers tokens through an onTextChunk callback while
  // prompt() runs. This bridges that callback into an async generator with
  // a promise queue: the callback pushes chunks and wakes the consumer,
  // the consumer drains the queue and sleeps until the next push.
  async *stream(
    alias: string,
    message: string | ChatMessage[],
    options: ChatOptions = {}
  ): AsyncGenerator<string> {
    const entry = this.requireLoaded(alias);
    const { systemPrompt, userText, priorHistory } = this.resolveMessage(message, options.system);

    const queue: Array<string | typeof STREAM_DONE> = [];
    let wake: (() => void) | null = null;
    let failure: Error | null = null;

    const push = (item: string | typeof STREAM_DONE): void => {
      queue.push(item);
      if (wake) {
        const resolve = wake;
        wake = null;
        resolve();
      }
    };

    const sequence = entry.context.getSequence();
    const session = new LlamaChatSession({
      contextSequence: sequence,
      systemPrompt,
    });

    if (priorHistory.length > 0) {
      session.setChatHistory(priorHistory);
    }

    // Kick off generation. Do not await here; the generator body below
    // consumes chunks while this runs.
    const generation = session
      .prompt(userText, {
        temperature: options.temperature ?? DEFAULT_TEMPERATURE,
        maxTokens: options.maxTokens ?? DEFAULT_MAX_TOKENS,
        onTextChunk: (chunk: string) => {
          push(chunk);
        },
      })
      .then(() => {
        push(STREAM_DONE);
      })
      .catch((err: unknown) => {
        failure = err instanceof Error ? err : new Error(String(err));
        push(STREAM_DONE);
      });

    try {
      while (true) {
        while (queue.length === 0) {
          await new Promise<void>((resolve) => {
            wake = resolve;
          });
        }

        const item = queue.shift() as string | typeof STREAM_DONE;
        if (item === STREAM_DONE) {
          break;
        }
        yield item;
      }

      await generation;
      if (failure) {
        throw failure;
      }
    } finally {
      sequence.dispose();
    }
  }

  // Create a Session that keeps conversation history for you.
  session(alias: string, options: { system?: string; temperature?: number } = {}): Session {
    this.requireLoaded(alias);
    return new Session(this, alias, options);
  }

  // Aliases of every model currently in memory.
  loaded(): string[] {
    return Array.from(this._loaded.keys());
  }

  has(alias: string): boolean {
    return this._loaded.has(alias);
  }

  private requireLoaded(alias: string): LoadedModel {
    const entry = this._loaded.get(alias);
    if (!entry) {
      throw new Error(
        `Model alias "${alias}" is not loaded. Loaded models: [${this.loaded().join(", ")}]. ` +
          `Load one with: await engine.load("${alias}", "/path/to/model.gguf")`
      );
    }
    return entry;
  }

  // Normalize the two accepted message shapes into a system prompt and a
  // user string. A plain string is the user message. An array of role and
  // content pairs uses the first system message and the last user message.
  private resolveMessage(
    message: string | ChatMessage[],
    systemOverride?: string
  ): { systemPrompt: string | undefined; userText: string; priorHistory: ChatHistoryItem[] } {
    if (typeof message === "string") {
      return { systemPrompt: systemOverride, userText: message, priorHistory: [] };
    }

    if (!Array.isArray(message) || message.length === 0) {
      throw new Error("Message must be a non-empty string or a non-empty array of {role, content} objects.");
    }

    const firstSystem = message.find((m) => m.role === "system");

    let lastUserIdx = -1;
    for (let i = message.length - 1; i >= 0; i--) {
      if (message[i].role === "user") { lastUserIdx = i; break; }
    }

    if (lastUserIdx === -1) {
      throw new Error("Message array must contain at least one message with role \"user\".");
    }

    const priorHistory: ChatHistoryItem[] = message
      .slice(0, lastUserIdx)
      .filter((m) => m.role !== "system")
      .map((m): ChatHistoryItem =>
        m.role === "user"
          ? { type: "user", text: m.content }
          : { type: "model", response: [m.content] }
      );

    return {
      systemPrompt: systemOverride ?? firstSystem?.content,
      userText: message[lastUserIdx].content,
      priorHistory,
    };
  }
}
