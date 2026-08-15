// 3th Floor AI SDK - Model Manager
// Author: Justin Bench, 3th Floor AI
//
// Tracks registered GGUF models in a local JSON registry and handles
// downloads from HuggingFace. No daemons, no ports.

import path from "path";
import fs from "fs";
import http from "http";
import https from "https";
import os from "os";
import { IncomingMessage } from "http";

export interface ModelEntry {
  alias: string;
  path: string;
  ctx: number;
  addedAt: number;
  sizeMb: number;
}

interface HuggingFaceSibling {
  rfilename: string;
}

interface HuggingFaceModelInfo {
  siblings: HuggingFaceSibling[];
}

// Strip extension(s) from a filename to produce a clean alias.
// "mistral-7b-instruct.Q4_K_M.gguf" -> "mistral-7b-instruct.Q4_K_M"
function fileStem(filename: string): string {
  return filename.replace(/\.gguf$/i, "");
}

// Pull bytes from a URL following redirects, writing to a stream.
// Resolves with the final response so the caller can read headers.
function fetchWithRedirects(url: string, dest: fs.WriteStream): Promise<void> {
  return new Promise((resolve, reject) => {
    function follow(target: string, hops: number): void {
      if (hops > 10) {
        reject(new Error("Too many redirects while downloading model."));
        return;
      }

      const transport = target.startsWith("http://") ? http : https;
      transport.get(target, (res: IncomingMessage) => {
        if (
          res.statusCode !== undefined &&
          res.statusCode >= 300 &&
          res.statusCode < 400 &&
          res.headers.location
        ) {
          follow(res.headers.location, hops + 1);
          return;
        }

        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} downloading ${target}`));
          return;
        }

        const total = parseInt(res.headers["content-length"] ?? "0", 10);
        let received = 0;
        let lastPct = -1;

        res.on("data", (chunk: Buffer) => {
          received += chunk.length;
          if (total > 0) {
            const pct = Math.floor((received / total) * 100);
            if (pct !== lastPct && pct % 5 === 0) {
              process.stdout.write(
                `\r  downloading... ${pct}% (${(received / 1_048_576).toFixed(1)} / ${(total / 1_048_576).toFixed(1)} MB)`
              );
              lastPct = pct;
            }
          } else {
            process.stdout.write(
              `\r  downloading... ${(received / 1_048_576).toFixed(1)} MB`
            );
          }
        });

        res.pipe(dest);

        dest.on("finish", () => {
          process.stdout.write("\n");
          resolve();
        });

        dest.on("error", reject);
        res.on("error", reject);
      });
    }

    follow(url, 0);
  });
}

// Fetch JSON from a URL and parse it. Follows redirects.
function fetchJson<T>(url: string): Promise<T> {
  return new Promise((resolve, reject) => {
    function follow(target: string, hops: number): void {
      if (hops > 10) {
        reject(new Error("Too many redirects fetching metadata."));
        return;
      }

      const transport = target.startsWith("http://") ? http : https;
      transport.get(target, { headers: { Accept: "application/json" } }, (res: IncomingMessage) => {
        if (
          res.statusCode !== undefined &&
          res.statusCode >= 300 &&
          res.statusCode < 400 &&
          res.headers.location
        ) {
          follow(res.headers.location, hops + 1);
          return;
        }

        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} fetching ${target}`));
          return;
        }

        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")) as T);
          } catch (err) {
            reject(new Error(`Failed to parse JSON from ${target}: ${String(err)}`));
          }
        });
        res.on("error", reject);
      });
    }

    follow(url, 0);
  });
}

export class ModelManager {
  private registryPath: string;
  private modelsDir: string;

  constructor(private engine: any, modelsDir?: string) {
    const dir = modelsDir
      ? path.resolve(modelsDir.replace(/^~/, os.homedir()))
      : path.join(os.homedir(), ".3thfloor", "models");

    fs.mkdirSync(dir, { recursive: true });
    this.modelsDir = dir;
    this.registryPath = path.join(dir, "registry.json");
  }

  private read(): ModelEntry[] {
    try {
      const raw = fs.readFileSync(this.registryPath, "utf8");
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed as ModelEntry[];
    } catch {
      return [];
    }
  }

  private write(entries: ModelEntry[]): void {
    fs.writeFileSync(this.registryPath, JSON.stringify(entries, null, 2), "utf8");
  }

  list(): ModelEntry[] {
    return this.read();
  }

  add(alias: string, filePath: string, ctx = 4096): ModelEntry {
    const resolved = path.resolve(filePath.replace(/^~/, os.homedir()));

    if (!fs.existsSync(resolved)) {
      throw new Error(`File not found: ${resolved}`);
    }

    const stat = fs.statSync(resolved);
    const sizeMb = parseFloat((stat.size / 1_048_576).toFixed(2));

    const entry: ModelEntry = {
      alias,
      path: resolved,
      ctx,
      addedAt: Date.now(),
      sizeMb,
    };

    const entries = this.read();
    const idx = entries.findIndex((e) => e.alias === alias);

    if (idx >= 0) {
      entries[idx] = entry;
    } else {
      entries.push(entry);
    }

    this.write(entries);
    return entry;
  }

  remove(alias: string): void {
    const entries = this.read();
    const idx = entries.findIndex((e) => e.alias === alias);

    if (idx < 0) {
      throw new Error(
        `Model alias "${alias}" not found in registry. ` +
          `Registered aliases: [${entries.map((e) => e.alias).join(", ")}]`
      );
    }

    entries.splice(idx, 1);
    this.write(entries);
  }

  info(alias: string): ModelEntry & { loaded: boolean } {
    const entries = this.read();
    const entry = entries.find((e) => e.alias === alias);

    if (!entry) {
      throw new Error(
        `Model alias "${alias}" not found in registry. ` +
          `Registered aliases: [${entries.map((e) => e.alias).join(", ")}]`
      );
    }

    return { ...entry, loaded: this.engine.has(alias) as boolean };
  }

  async download(
    repoId: string,
    options: { filename?: string; quantization?: string } = {}
  ): Promise<string> {
    let filename = options.filename;

    if (!filename) {
      const quant = options.quantization ?? "Q4_K_M";
      const apiUrl = `https://huggingface.co/api/models/${repoId}`;

      process.stdout.write(`  fetching file listing for ${repoId}...\n`);

      const meta = await fetchJson<HuggingFaceModelInfo>(apiUrl);

      if (!meta.siblings || meta.siblings.length === 0) {
        throw new Error(
          `No files found in HuggingFace repo "${repoId}". ` +
            `Check the repo ID and make sure it is public.`
        );
      }

      const ggufFiles = meta.siblings
        .map((s) => s.rfilename)
        .filter((n) => n.toLowerCase().endsWith(".gguf"));

      if (ggufFiles.length === 0) {
        throw new Error(
          `No GGUF files found in repo "${repoId}". ` +
            `Available files: ${meta.siblings.map((s) => s.rfilename).join(", ")}`
        );
      }

      // Find one matching the requested quantization, case-insensitive.
      const match = ggufFiles.find((n) =>
        n.toUpperCase().includes(quant.toUpperCase())
      );

      if (!match) {
        throw new Error(
          `No GGUF file matching quantization "${quant}" found in repo "${repoId}". ` +
            `Available GGUF files: ${ggufFiles.join(", ")}`
        );
      }

      filename = match;
    }

    const destPath = path.join(this.modelsDir, filename);

    if (fs.existsSync(destPath)) {
      process.stdout.write(`  file already exists at ${destPath}, skipping download.\n`);
    } else {
      const downloadUrl = `https://huggingface.co/${repoId}/resolve/main/${filename}`;
      process.stdout.write(`  downloading ${filename} from ${repoId}...\n`);

      const tmpPath = `${destPath}.part`;
      const dest = fs.createWriteStream(tmpPath);

      try {
        await fetchWithRedirects(downloadUrl, dest);
        fs.renameSync(tmpPath, destPath);
      } catch (err) {
        // Clean up partial file on failure.
        try { fs.unlinkSync(tmpPath); } catch { /* already gone */ }
        throw err;
      }
    }

    const alias = fileStem(filename);
    this.add(alias, destPath);

    process.stdout.write(`  registered as alias "${alias}".\n`);
    return destPath;
  }
}
