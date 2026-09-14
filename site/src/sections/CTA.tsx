import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { DOCS_URL, REPO_URL } from "../lib/links";

export default function CTA() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section ref={ref} className="relative py-24 md:py-32">
      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="relative overflow-hidden rounded-[28px] border border-white/[0.07] bg-gradient-to-b from-ink-800/70 to-ink-900/70 px-8 py-16 md:px-16 md:py-24 text-center shadow-glow"
        >
          <div aria-hidden className="absolute inset-0 -z-10">
            <div className="absolute inset-0 dotgrid opacity-40" />
            <div className="absolute left-1/2 top-0 h-[60%] w-[80%] -translate-x-1/2 rounded-full bg-[radial-gradient(ellipse_at_top,rgba(124,92,255,0.35),transparent_60%)] blur-3xl" />
            <div className="absolute right-0 bottom-0 h-[50%] w-[60%] rounded-full bg-[radial-gradient(ellipse_at_bottom_right,rgba(94,226,255,0.18),transparent_60%)] blur-3xl" />
          </div>

          <span className="pill">Ship it</span>
          <h2 className="mt-6 text-balance text-[36px] md:text-[56px] font-semibold tracking-tightest leading-[1.02] gradient-text">
            Give your agent a memory
            <br className="hidden md:block" /> that gets sharper with use.
          </h2>
          <p className="mx-auto mt-6 max-w-xl text-base md:text-lg text-chalk-400 leading-relaxed">
            One <code className="font-mono text-chalk-200">pip install</code>{" "}
            and you&apos;re wired up. Open source under Apache-2.0, paper-faithful,
            and quietly opinionated.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-3">
            <a href={DOCS_URL} target="_blank" rel="noreferrer" className="btn-primary">
              Install methodos
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14M13 5l7 7-7 7" />
              </svg>
            </a>
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="btn-ghost">
              View source
            </a>
          </div>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-[12px] text-chalk-500">
            <span className="flex items-center gap-2"><Dot /> Apache-2.0</span>
            <span className="flex items-center gap-2"><Dot /> Python 3.12+</span>
            <span className="flex items-center gap-2"><Dot /> FastAPI · Pydantic v2</span>
            <span className="flex items-center gap-2"><Dot /> SQLite · sqlite-vec</span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function Dot() {
  return <span className="inline-block h-1 w-1 rounded-full bg-chalk-500" />;
}
