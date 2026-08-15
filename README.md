# @3thfloor/engine

Local AI inference for Node.js and Electron. One install. No daemon. No API keys.

Ask a question, get a string back. That is the whole API surface you need to learn.

```js
const answer = await engine.chat("assistant", "What is a smoke test?");
console.log(answer); // a plain string
```

No `.choices[0].message.content`. No response envelopes. No server to babysit. The model is an object in memory that lives and dies with your process.

## Install

```bash
npm install @3thfloor/engine
```

The install downloads the right prebuilt native binary for your platform automatically (macOS arm64/x64, Linux x64, Windows x64). No compiler toolchain required. GPU acceleration (Metal on macOS, CUDA and Vulkan on Linux/Windows) is detected and used when available.

Requires Node 18 or later.

## Quick Start

TypeScript:

```ts
import { Engine } from "@3thfloor/engine";

const engine = new Engine();
await engine.load("assistant", "./models/qwen3-4b-q4.gguf");

const answer: string = await engine.chat("assistant", "Write one unit test case name for a login form.");
console.log(answer);
```

JavaScript (ESM):

```js
import { Engine } from "@3thfloor/engine";

const engine = new Engine();
await engine.load("assistant", "./models/qwen3-4b-q4.gguf");

const answer = await engine.chat("assistant", "Write one unit test case name for a login form.");
console.log(answer);
```

`load()` gives the model an alias. Every other call refers to the model by that alias. `chat()` returns a string. Done.

## Sessions (Conversation History)

A `Session` keeps conversation history for you. Each `send()` includes everything said so far.

```ts
import { Engine } from "@3thfloor/engine";

const engine = new Engine();
await engine.load("assistant", "./models/qwen3-4b-q4.gguf");

const session = engine.session("assistant", {
  system: "You are a QA lead. Answer in two sentences or fewer.",
});

const first = await session.send("What is boundary value analysis?");
console.log(first);

const second = await session.send("Give me one example using an age field.");
console.log(second); // the model remembers the first exchange
```

Call `session.clear()` to clear history, or `session.history` to inspect it as an array of `{ role, content }` objects.

## Streaming

`stream()` returns an async iterator of token strings. Print them as they arrive:

```ts
for await (const token of engine.stream("assistant", "Explain regression testing in one paragraph.")) {
  process.stdout.write(token);
}
process.stdout.write("\n");
```

Sessions stream too:

```ts
for await (const token of session.stream("Now explain it to a new hire.")) {
  process.stdout.write(token);
}
```

## Multiple Models

Load as many models as your memory allows. Route between them by alias:

```ts
const engine = new Engine();
await engine.load("fast", "./models/qwen3-4b-q4.gguf");
await engine.load("smart", "./models/qwen3-32b-q4.gguf");

// Small model triages, big model handles the hard stuff
const triage = await engine.chat("fast", `Is this question simple or complex? Answer one word.\n\n${question}`);
const alias = triage.toLowerCase().includes("complex") ? "smart" : "fast";

const answer = await engine.chat(alias, question);
```

Unload a model when you are done with it:

```ts
await engine.unload("fast");
```

## Agents

Define tools as plain functions, then let the model call them:

```ts
import { Engine, tool, runAgent } from "@3thfloor/engine";

const engine = new Engine();
await engine.load("assistant", "./models/qwen3-32b-q4.gguf");

const getTestCount = tool(
  "get_test_count",
  "Returns the number of tests in the suite",
  async () => {
    return { total: 412, passing: 398, failing: 14 };
  }
);

const result = await runAgent(engine, "assistant", "How many tests are failing right now?", {
  tools: [getTestCount],
});

console.log(result); // a string, same as chat()
```

The agent loop handles tool selection, execution, and feeding results back to the model. You get the final answer as a string.

## Model Management

The engine ships with a model registry so you are not passing file paths around your codebase:

```ts
// Register a local file under an alias
engine.models.add("qwen-4b", "./models/qwen3-4b-q4.gguf");

// See what is registered
console.log(engine.models.list());
// [ { alias: "qwen-4b", path: "/abs/path/to/qwen3-4b-q4.gguf", sizeMb: 2548.3, ctx: 4096, ... } ]

// Download straight from HuggingFace (auto-picks Q4_K_M by default)
await engine.models.download("Qwen/Qwen3-4B-Instruct-GGUF");

// Or request a specific quantization
await engine.models.download("Qwen/Qwen3-4B-Instruct-GGUF", { quantization: "Q4_K_M" });

// Or a specific filename
await engine.models.download("Qwen/Qwen3-4B-Instruct-GGUF", { filename: "qwen3-4b-instruct-q4_k_m.gguf" });

// Load by registered alias (alias = filename without .gguf)
const entry = engine.models.info("qwen3-4b-instruct-q4_k_m");
await engine.load("assistant", entry.path);
```

Downloads go to `~/.3thfloor/models/` and are skipped if the file already exists.

## Use in Electron

The engine runs inside the Electron main process. No daemon, no localhost port, no HTTP. Expose it to your renderer with a single IPC handler:

```ts
// main.ts
import { app, ipcMain } from "electron";
import { Engine } from "@3thfloor/engine";

const engine = new Engine();

app.whenReady().then(async () => {
  await engine.load("assistant", "./models/qwen3-4b-q4.gguf");

  ipcMain.handle("ai:chat", async (_event, message: string) => {
    return engine.chat("assistant", message);
  });
});
```

```ts
// renderer (through your preload bridge)
const answer = await window.api.invoke("ai:chat", "Summarize this bug report.");
```

Because inference happens in the main process, your app ships as one binary with no background services to install, no ports to fight over, and nothing left running when the user quits.

## Use in Any Node App

The engine works anywhere Node 18+ runs: Express and Fastify APIs, CLI tools, test runners, cron jobs, queue workers. Create the engine once at startup, load your models, and the engine stays alive for the lifetime of your process. Every request after the first load is answered from the model already sitting in memory, so there is no cold start and no connection handling to write.

```ts
// express example
app.post("/ask", async (req, res) => {
  const answer = await engine.chat("assistant", req.body.question);
  res.json({ answer });
});
```

---

## License

Free for personal projects, research, experiments, and noncommercial use under the [PolyForm Noncommercial License 1.0.0](./LICENSE).

If you are shipping a product with this engine embedded, a commercial license is required. Contact [justin@3thfloor.com](mailto:justin@3thfloor.com).

---

Built by Justin Bench, [3th Floor AI](https://3thfloor.com).
