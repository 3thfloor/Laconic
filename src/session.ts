import type { Engine } from "./engine.js"

interface Message {
  role: "system" | "user" | "assistant"
  content: string
}

export class Session {
  private _engine: Engine
  private _alias: string
  private _temperature: number
  private _messages: Message[] = []

  constructor(
    engine: Engine,
    alias: string,
    options: { system?: string; temperature?: number } = {}
  ) {
    this._engine = engine
    this._alias = alias
    this._temperature = options.temperature ?? 0.7
    if (options.system) {
      this._messages.push({ role: "system", content: options.system })
    }
  }

  async send(message: string, options: { maxTokens?: number } = {}): Promise<string> {
    this._messages.push({ role: "user", content: message })

    const response = await this._engine.chat(
      this._alias,
      this._messages,
      {
        temperature: this._temperature,
        maxTokens: options.maxTokens,
      }
    )

    this._messages.push({ role: "assistant", content: response })

    return response
  }

  async *stream(
    message: string,
    options: { maxTokens?: number } = {}
  ): AsyncGenerator<string> {
    this._messages.push({ role: "user", content: message })

    const tokens = this._engine.stream(
      this._alias,
      this._messages,
      {
        temperature: this._temperature,
        maxTokens: options.maxTokens,
      }
    )

    let full = ""
    for await (const token of tokens) {
      full += token
      yield token
    }

    this._messages.push({ role: "assistant", content: full })
  }

  clear(): Session {
    this._messages = this._messages.filter(m => m.role === "system")
    return this
  }

  get history(): Message[] {
    return this._messages.filter(m => m.role !== "system")
  }

  get system(): string | undefined {
    return this._messages.find(m => m.role === "system")?.content
  }
}
