import { motion, useInView } from "framer-motion";
import { useRef } from "react";

type Feature = {
  title: string;
  desc: string;
  Icon: () => JSX.Element;
  span?: string;
};

const features: Feature[] = [
  {
    title: "Queryable procedural graph",
    desc:
      "Nodes, edges, and attributes are first-class. Match → extract → translate in milliseconds, with cached results keyed by graph, action, and observation.",
    Icon: IconGraph,
    span: "md:col-span-2",
  },
  {
    title: "Self-evolving",
    desc: "Edits are proposed offline, validated structurally, and accepted only if validation does not regress.",
    Icon: IconSpark,
  },
  {
    title: "Drop-in protocol",
    desc: "Solver is a single async step. Bring any ReAct agent. No rewrites.",
    Icon: IconPlug,
  },
  {
    title: "Typed end-to-end",
    desc: "Pydantic v2 + Protocol polymorphism. mypy --strict clean.",
    Icon: IconShield,
  },
  {
    title: "Local-first",
    desc: "SQLite by default. sqlite-vec for fuzzy match. FastAPI on 127.0.0.1.",
    Icon: IconDisk,
    span: "md:col-span-2",
  },
];

export default function Features() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="features" ref={ref} className="relative py-24 md:py-32">
      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-2xl"
        >
          <span className="pill">Features</span>
          <h2 className="mt-5 text-[32px] md:text-[48px] font-semibold tracking-tightest leading-[1.05] gradient-text">
            Built for production. Designed for clarity.
          </h2>
          <p className="mt-5 text-base md:text-lg text-chalk-400 leading-relaxed">
            Every feature earns its place. No semi-private names. No hidden
            side effects. No magic — just composable primitives that fit how
            agents already work.
          </p>
        </motion.div>

        <div className="mt-14 grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-5">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 24 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.7, delay: 0.05 + i * 0.06, ease: [0.22, 1, 0.36, 1] }}
              className={`group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-gradient-to-b from-ink-800/40 to-ink-900/40 p-7 transition-colors hover:border-white/[0.12] ${
                f.span ?? ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-white/[0.08] bg-white/[0.02] text-chalk-100">
                  <f.Icon />
                </span>
                <h3 className="text-[17px] font-semibold tracking-tight text-chalk-50">
                  {f.title}
                </h3>
              </div>
              <p className="mt-4 text-[14px] leading-relaxed text-chalk-400 max-w-md">
                {f.desc}
              </p>

              <div
                aria-hidden
                className="pointer-events-none absolute -right-10 -bottom-10 h-40 w-40 rounded-full bg-accent-500/0 blur-2xl transition-all duration-700 group-hover:bg-accent-500/20"
              />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function IconGraph() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
      <circle cx="6" cy="7" r="2" />
      <circle cx="18" cy="7" r="2" />
      <circle cx="12" cy="17" r="2" />
      <path d="M8 7h8M7.5 8.5l3 7M16.5 8.5l-3 7" />
    </svg>
  );
}
function IconSpark() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
      <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.5 5.5l2.8 2.8M15.7 15.7l2.8 2.8M5.5 18.5l2.8-2.8M15.7 8.3l2.8-2.8" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}
function IconPlug() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 2v6M15 2v6M6 8h12v4a6 6 0 0 1-12 0z" />
      <path d="M12 18v4" />
    </svg>
  );
}
function IconShield() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}
function IconDisk() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="6" rx="8" ry="3" />
      <path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
    </svg>
  );
}
