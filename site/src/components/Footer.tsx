import Logo from "./Logo";
import {
  ARCH_URL,
  API_URL,
  PAPER_URL,
  REPO_URL,
  ISSUES_URL,
} from "../lib/links";

const groups = [
  {
    title: "Product",
    items: [
      { label: "Install", href: REPO_URL },
      { label: "Examples", href: `${REPO_URL}/tree/master/examples` },
      { label: "Roadmap", href: `${REPO_URL}/issues` },
    ],
  },
  {
    title: "Reference",
    items: [
      { label: "Architecture", href: ARCH_URL },
      { label: "API", href: API_URL },
      { label: "Paper mapping", href: PAPER_URL },
    ],
  },
  {
    title: "Project",
    items: [
      { label: "GitHub", href: REPO_URL },
      { label: "Issues", href: ISSUES_URL },
      { label: "Apache-2.0", href: `${REPO_URL}/blob/master/LICENSE` },
    ],
  },
];

export default function Footer() {
  return (
    <footer className="relative border-t border-white/[0.06]">
      <div className="container-x py-16">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-10">
          <div className="col-span-2 md:col-span-2">
            <div className="flex items-center gap-2.5">
              <Logo className="h-7 w-7" />
              <span className="text-[15px] font-semibold tracking-tight text-chalk-50">
                methodos
              </span>
            </div>
            <p className="mt-4 max-w-xs text-[13px] text-chalk-400 leading-relaxed">
              A self-evolving procedural graph adapter for LLM agents.
              Open source. Paper-faithful. Production-minded.
            </p>
          </div>

          {groups.map((g) => (
            <div key={g.title}>
              <div className="text-[11px] uppercase tracking-[0.2em] text-chalk-500">
                {g.title}
              </div>
              <ul className="mt-4 space-y-2.5">
                {g.items.map((it) => (
                  <li key={it.label}>
                    <a
                      href={it.href}
                      target="_blank"
                      rel="noreferrer"
                      className="text-[13px] text-chalk-300 hover:text-chalk-50 transition-colors"
                    >
                      {it.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="hairline my-12" />

        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="text-[12px] text-chalk-500">
            © {new Date().getFullYear()} methodos · Apache-2.0
          </div>
          <div className="text-[12px] text-chalk-500 font-mono">
            pip install methodos
          </div>
        </div>
      </div>
    </footer>
  );
}
