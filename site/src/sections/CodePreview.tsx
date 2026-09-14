import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { DOCS_URL } from "../lib/links";

const code = `from methodos import (
    Node, Edge, Attribute, Relation,
    ProceduralGraph, PGAdapter, Solver, AgentState,
    LiteLLMClient,
)


class MySolver:
    """Any async step(state) -> str satisfies the Solver Protocol."""

    async def step(self, state: AgentState) -> str:
        # state.context already contains the procedural guidance.
        return "search"


graph = ProceduralGraph(
    id="example",
    nodes={"start": Node(id="start"),
           "search": Node(id="search", description="search Wikipedia"),
           "answer": Node(id="answer")},
    edges=[
        Edge("start", "search", Relation.LEADS_TO,
             Attribute(condition="need information",
                       guidance="call search with focused query",
                       pitfalls="don't search with the full question")),
        Edge("search", "answer", Relation.LEADS_TO,
             Attribute(condition="have enough info",
                       guidance="synthesize a concise answer",
                       pitfalls="don't repeat observations")),
    ],
    terminal_ids={"answer"},
)

adapter = PGAdapter(
    solver=MySolver(),
    graph=graph,
    llm=LiteLLMClient(model="gpt-4o-mini"),
)

action = await adapter.step(query="What is the capital of France?",
                            trajectory=trajectory)`;

export default function CodePreview() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section ref={ref} className="relative py-24 md:py-32 overflow-hidden">
      <div className="absolute inset-0 -z-10">
        <div className="absolute left-1/2 top-1/2 h-[60vh] w-[80vw] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(ellipse_at_center,rgba(124,92,255,0.18),transparent_60%)] blur-3xl" />
      </div>

      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-2xl"
        >
          <span className="pill">Code · it&apos;s small</span>
          <h2 className="mt-5 text-[32px] md:text-[48px] font-semibold tracking-tightest leading-[1.05] gradient-text">
            Drop in. Wrap. Forget it&apos;s there.
          </h2>
          <p className="mt-5 text-base md:text-lg text-chalk-400 leading-relaxed">
            A minimal example that runs end-to-end: any solver, any graph, any
            LiteLLM-supported model. The adapter handles matching, guidance,
            and caching — you write the agent.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.9, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
          className="relative mt-14 overflow-hidden rounded-3xl border border-white/[0.07] bg-ink-900/70 shadow-glow"
        >
          <div className="flex items-center justify-between border-b border-white/[0.05] px-5 py-3">
            <div className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
              <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
              <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
            </div>
            <div className="font-mono text-[11px] text-chalk-400">examples/simple_react.py</div>
            <div className="flex items-center gap-2">
              <span className="rounded-full border border-white/10 bg-white/[0.02] px-2 py-0.5 text-[10px] text-chalk-400">
                python 3.12
              </span>
            </div>
          </div>

          <pre className="relative overflow-x-auto p-6 md:p-8 text-[12.5px] md:text-[13px] leading-[1.7] font-mono text-chalk-200">
            <code dangerouslySetInnerHTML={{ __html: highlight(code) }} />
          </pre>

          <div className="border-t border-white/[0.05] bg-ink-950/40 px-5 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div className="font-mono text-[12px] text-chalk-400">
              <span className="text-chalk-500">$</span> pip install methodos
            </div>
            <a href={DOCS_URL} target="_blank" rel="noreferrer" className="btn-ghost !py-2 text-[12.5px]">
              Full API reference
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14M13 5l7 7-7 7" />
              </svg>
            </a>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Minimal hand-rolled highlighter (no external deps).
function highlight(src: string): string {
  const escape = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // Tokenize strings first to avoid highlighting inside them.
  const tokens: { type: string; text: string }[] = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === '"' || ch === "'") {
      const quote = ch;
      let j = i + 1;
      while (j < src.length && src[j] !== quote) j++;
      tokens.push({ type: "str", text: src.slice(i, j + 1) });
      i = j + 1;
      continue;
    }
    if (ch === "#") {
      let j = i;
      while (j < src.length && src[j] !== "\n") j++;
      tokens.push({ type: "com", text: src.slice(i, j) });
      i = j;
      continue;
    }
    let j = i;
    while (
      j < src.length &&
      src[j] !== '"' &&
      src[j] !== "'" &&
      src[j] !== "#" &&
      /[A-Za-z0-9_]/.test(src[j])
    ) {
      j++;
    }
    if (j > i) {
      tokens.push({ type: "word", text: src.slice(i, j) });
      i = j;
      continue;
    }
    tokens.push({ type: "raw", text: ch });
    i++;
  }

  const KW = new Set([
    "from","import","class","async","def","return","if","else","elif","for","while","in","not","and","or","with","as","raise","try","except","finally","pass","None","True","False","is",
  ]);
  const CLS = new Set([
    "Node","Edge","Attribute","Relation","ProceduralGraph","PGAdapter","Solver","AgentState","LiteLLMClient","MySolver",
  ]);

  return tokens
    .map((t) => {
      const esc = escape(t.text);
      if (t.type === "str") return `<span class="t-str">${esc}</span>`;
      if (t.type === "com") return `<span class="t-com">${esc}</span>`;
      if (t.type === "raw") return esc;
      if (/^\d+$/.test(t.text)) return `<span class="t-num">${esc}</span>`;
      if (KW.has(t.text)) return `<span class="t-kw">${esc}</span>`;
      if (CLS.has(t.text)) return `<span class="t-cls">${esc}</span>`;
      return esc;
    })
    .join("");
}
