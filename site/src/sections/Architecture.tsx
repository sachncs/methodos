import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { ARCH_URL } from "../lib/links";

const modules = [
  { name: "schema.py", desc: "Pydantic v2 models + Edit discriminated union", accent: "from-accent-500/40" },
  { name: "graph.py", desc: "Pure functions: match · neighborhood · validate · apply_edits", accent: "from-accent-500/30" },
  { name: "llm.py", desc: "LLMClient Protocol + LiteLLMClient", accent: "from-accent-500/25" },
  { name: "guidance.py", desc: "Ψ prompt + generate_guidance (cached)", accent: "from-accent-500/20" },
  { name: "adapter.py", desc: "AgentState · Solver Protocol · PGAdapter", accent: "from-electric/30" },
  { name: "evolution.py", desc: "Algorithm 1 · EvolutionEngine", accent: "from-electric/25" },
  { name: "repo.py", desc: "Repository + VectorIndex (SQLite, FS, sqlite-vec)", accent: "from-electric/20" },
  { name: "service.py", desc: "FastAPI app factory (REST: /v1/graphs, /guidance)", accent: "from-electric/15" },
  { name: "cli.py", desc: "Typer CLI · init · inspect · serve · replay · eval", accent: "from-white/15" },
];

export default function Architecture() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="architecture" ref={ref} className="relative py-24 md:py-32">
      <div className="container-x">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="lg:col-span-5"
          >
            <span className="pill">Architecture</span>
            <h2 className="mt-5 text-[32px] md:text-[48px] font-semibold tracking-tightest leading-[1.05] gradient-text">
              Nine files. One coherent loop.
            </h2>
            <p className="mt-5 text-base md:text-lg text-chalk-400 leading-relaxed">
              No sprawling framework. No rebrand of a popular library.
              methodos is a single Python package with nine focused modules —
              each one a clear unit of behavior, with public APIs you can read
              end-to-end.
            </p>

            <div className="mt-8 rounded-2xl border border-white/[0.06] bg-ink-900/40 p-6">
              <div className="text-[11px] uppercase tracking-[0.2em] text-chalk-500">
                Standards
              </div>
              <ul className="mt-4 space-y-3">
                {[
                  "Google Python Style",
                  "typing.Protocol everywhere",
                  "Top-of-file absolute imports",
                  "mypy --strict clean",
                  "≥ 95% test coverage",
                  "No semi-private names",
                ].map((s) => (
                  <li key={s} className="flex items-center gap-3 text-[13.5px] text-chalk-200">
                    <Check />
                    {s}
                  </li>
                ))}
              </ul>
              <a
                href={ARCH_URL}
                target="_blank"
                rel="noreferrer"
                className="mt-6 inline-flex items-center gap-2 text-[13px] text-chalk-300 hover:text-chalk-50"
              >
                Read architecture.md
                <Arrow />
              </a>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.9, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
            className="lg:col-span-7"
          >
            <div className="relative overflow-hidden rounded-3xl border border-white/[0.06] bg-gradient-to-b from-ink-800/50 to-ink-900/50 shadow-ring">
              <div className="flex items-center justify-between border-b border-white/[0.05] px-5 py-3">
                <div className="font-mono text-[11px] text-chalk-500">src/methodos/</div>
                <div className="font-mono text-[11px] text-chalk-500">9 modules</div>
              </div>
              <div className="divide-y divide-white/[0.04]">
                {modules.map((m, i) => (
                  <motion.div
                    key={m.name}
                    initial={{ opacity: 0, x: -8 }}
                    animate={inView ? { opacity: 1, x: 0 } : {}}
                    transition={{ duration: 0.5, delay: 0.2 + i * 0.04 }}
                    className="group relative grid grid-cols-12 items-center gap-4 px-5 py-4 transition-colors hover:bg-white/[0.025]"
                  >
                    <span
                      aria-hidden
                      className={`absolute left-0 top-0 h-full w-px bg-gradient-to-b ${m.accent} to-transparent opacity-0 group-hover:opacity-100 transition-opacity`}
                    />
                    <div className="col-span-4 font-mono text-[13px] text-chalk-100">
                      {m.name}
                    </div>
                    <div className="col-span-8 text-[13px] text-chalk-400">
                      {m.desc}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function Check() {
  return (
    <span className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-white/10 bg-white/[0.02]">
      <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-accent-300">
        <path d="M20 6L9 17l-5-5" />
      </svg>
    </span>
  );
}
function Arrow() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14M13 5l7 7-7 7" />
    </svg>
  );
}
