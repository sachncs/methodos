type Props = { className?: string };

export default function Logo({ className }: Props) {
  return (
    <svg
      viewBox="0 0 64 64"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden
    >
      <defs>
        <linearGradient id="lg-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#A88EFF" />
          <stop offset="100%" stopColor="#5EE2FF" />
        </linearGradient>
        <linearGradient id="lg-bg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#15161F" />
          <stop offset="100%" stopColor="#0A0B10" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="url(#lg-bg)" />
      <g stroke="url(#lg-g)" strokeWidth="1.6" fill="none" strokeLinecap="round">
        <path d="M16 20 L48 20" opacity="0.55" />
        <path d="M32 20 L32 46" opacity="0.55" />
        <path d="M20 46 L44 46" opacity="0.35" />
        <path d="M22 32 L42 32" opacity="0.35" />
      </g>
      <g>
        <circle cx="16" cy="20" r="3.4" fill="#A88EFF" />
        <circle cx="48" cy="20" r="3.4" fill="#5EE2FF" />
        <circle cx="32" cy="46" r="3.4" fill="#FFFFFF" />
        <circle cx="22" cy="32" r="2.4" fill="#7C5CFF" />
        <circle cx="42" cy="32" r="2.4" fill="#5EE2FF" />
      </g>
    </svg>
  );
}
