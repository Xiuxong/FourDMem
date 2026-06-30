/**
 * FourDMem Auto-Archive Hook for Oh My Pi
 *
 * Captures tool results during each turn and archives them to FourDMem L0
 * at turn end. Also triggers L0→L1 extraction and checkpoint.
 *
 * Discovery: Oh My Pi loads this from `.omp/hooks/` automatically.
 * Claude Code uses `.claude-plugin/hooks/` (separate hook).
 */
import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";

// ── Project root resolution ────────────────────────────────────────────────

function findProjectRoot(startDir: string): string | null {
  let dir = startDir;
  for (let i = 0; i < 10; i++) {
    if (existsSync(join(dir, "Cargo.toml"))) return dir;
    const parent = resolve(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

// ── Python resolution ──────────────────────────────────────────────────────

function findPython(projectRoot: string): string | null {
  const candidates = [
    join(projectRoot, "python", ".venv", "Scripts", "python.exe"),
    join(projectRoot, "python", ".venv", "bin", "python"),
    join(projectRoot, ".venv", "Scripts", "python.exe"),
    join(projectRoot, ".venv", "bin", "python"),
    "python",
    "python3",
  ];
  for (const candidate of candidates) {
    try {
      execSync(`"${candidate}" --version`, { stdio: "ignore", timeout: 3000 });
      return candidate;
    } catch {
      continue;
    }
  }
  return null;
}

// ── Buffer for tool results within a turn ──────────────────────────────────

interface CapturedOutput {
  toolName: string;
  content: string;
  timestamp: number;
}

const turnBuffer: CapturedOutput[] = [];
let turnActive = false;

// ── Archive to FourDMem ────────────────────────────────────────────────────

function archiveToFourDmem(
  projectRoot: string,
  python: string,
  outputs: CapturedOutput[]
): void {
  if (outputs.length === 0) return;

  const dbPath = join(projectRoot, "data", "vault", "evidence.db");
  if (!existsSync(dbPath)) return;

  // Build archive script inline — avoids dependency on stop-hook format
  const archiveScript = `
import sys, json, uuid
sys.path.insert(0, ${JSON.stringify(join(projectRoot, "python"))})

try:
    import fourdmem
    from cognition.embed_utils import ingest_safely

    engine = fourdmem.FourDMemEngine(${JSON.stringify(dbPath)})
    engine.advance_tick()

    session_id = "omp-" + uuid.uuid4().hex[:8]
    outputs = json.loads(${JSON.stringify(JSON.stringify(outputs))})

    # Archive each tool result to L0
    for item in outputs:
        content = item["content"][:3000]
        if len(content.strip()) < 10:
            continue
        meta = json.dumps({
            "source": "omp_hook",
            "tool": item["toolName"],
            "turn_type": "tool_result",
        })
        ingest_safely(engine, session_id, "tool", content, meta, "python")

    # L0→L1 extraction from recent evidence
    from cognition.extractor import FactExtractor
    from cognition.dedup import SemanticDeduplicator

    import sqlite3
    conn = sqlite3.connect(${JSON.stringify(dbPath)})
    rows = conn.execute(
        "SELECT id, role, content FROM evidence ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()

    if len(rows) >= 2:
        facts = FactExtractor().extract_from_evidence(
            [{"id": r[0], "role": r[1], "content": r[2]} for r in rows]
        )
        if facts:
            dedup = SemanticDeduplicator()
            n = 0
            for f in facts[:8]:
                label = f.get("label", "")
                if label and len(label.strip()) >= 8:
                    result = dedup.add_fact_with_dedup(
                        engine, label, f.get("l0_refs", [])
                    )
                    if result.get("status") in ("added", "merged"):
                        n += 1

    # Checkpoint to disk
    engine.checkpoint(${JSON.stringify(join(projectRoot, "data", "vault"))})
except Exception as e:
    pass
`;

  try {
    execSync(`"${python}" -c ${JSON.stringify(archiveScript)}`, {
      cwd: projectRoot,
      timeout: 30_000,
      stdio: "pipe",
    });
  } catch {
    // Silent fail — don't break the agent loop
  }
}

// ── Hook entry point ───────────────────────────────────────────────────────

export default function fourdmemAutoArchive(omp: {
  on: (event: string, handler: (event: unknown, ctx: unknown) => Promise<unknown>) => void;
}): void {
  // Capture tool results during the turn
  omp.on("tool_result", async (event: unknown, _ctx: unknown) => {
    if (!turnActive) return;

    const e = event as Record<string, unknown>;
    const toolName = typeof e.toolName === "string" ? e.toolName : "unknown";
    const isError = typeof e.isError === "boolean" ? e.isError : false;

    // Skip errored results and internal tools
    if (isError) return;
    if (["search_memory", "submit_feedback", "wake_up"].includes(toolName)) return;

    // Extract text content
    const content = Array.isArray(e.content)
      ? e.content
          .filter((c: unknown): c is Record<string, unknown> =>
            typeof c === "object" && c !== null && "type" in c
          )
          .filter((c: Record<string, unknown>) => c.type === "text")
          .map((c: Record<string, unknown>) =>
            typeof c.text === "string" ? c.text : ""
          )
          .join("\n")
      : "";

    if (content.length > 10) {
      turnBuffer.push({
        toolName,
        content,
        timestamp: Date.now(),
      });
    }
  });

  // Start of turn — begin buffering
  omp.on("turn_start", async (_event: unknown, _ctx: unknown) => {
    turnActive = true;
    turnBuffer.length = 0;
  });

  // End of turn — archive buffered content
  omp.on("turn_end", async (_event: unknown, _ctx: unknown) => {
    turnActive = false;

    if (turnBuffer.length === 0) return;

    const hookDir = __dirname;
    const projectRoot = findProjectRoot(hookDir);
    if (!projectRoot) return;

    const python = findPython(projectRoot);
    if (!python) return;

    // Snapshot and clear buffer
    const snapshot = [...turnBuffer];
    turnBuffer.length = 0;

    // Archive in background (don't block agent)
    archiveToFourDmem(projectRoot, python, snapshot);
  });

  // Session shutdown — final archive
  omp.on("session_shutdown", async (_event: unknown, _ctx: unknown) => {
    if (turnBuffer.length === 0) return;

    const hookDir = __dirname;
    const projectRoot = findProjectRoot(hookDir);
    if (!projectRoot) return;

    const python = findPython(projectRoot);
    if (!python) return;

    archiveToFourDmem(projectRoot, python, [...turnBuffer]);
    turnBuffer.length = 0;
  });
}
