import { useId } from "react";

/** Original artwork for the launcher. These are illustrations, not generated model output. */
export function SceneArt({
  kind = "redwood",
  className = "",
}: {
  kind?: string;
  className?: string;
}) {
  const uid = useId().replace(/:/g, "");
  const forest = kind === "redwood" || kind === "horror";
  const road = kind === "road";
  const alien = kind === "alien";
  const dungeon = kind === "dungeon";
  const hue = forest
    ? ["#101e20", "#688b72", "#090f13"]
    : road
      ? ["#151c3c", "#e39481", "#111626"]
      : alien
        ? ["#192835", "#b2bd9f", "#101b25"]
        : dungeon
          ? ["#252023", "#d49c5b", "#111016"]
          : ["#392952", "#e5b0a1", "#15192a"];
  return (
    <svg
      className={`scene-art ${className}`}
      viewBox="0 0 1200 750"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={`${uid}s`} x2="0" y2="1">
          <stop stopColor={hue[0]} />
          <stop offset=".56" stopColor={hue[1]} />
          <stop offset="1" stopColor={hue[2]} />
        </linearGradient>
        <radialGradient id={`${uid}g`}>
          <stop stopColor="#ebf4ce" stopOpacity=".56" />
          <stop offset="1" stopColor="#bdd6b1" stopOpacity="0" />
        </radialGradient>
        <linearGradient id={`${uid}f`} x2="0" y2="1">
          <stop stopColor="#cbd6b3" stopOpacity="0" />
          <stop offset="1" stopColor="#bad0b4" stopOpacity=".17" />
        </linearGradient>
        <filter id={`${uid}b`}>
          <feGaussianBlur stdDeviation="18" />
        </filter>
        <filter id={`${uid}t`}>
          <feTurbulence
            type="fractalNoise"
            baseFrequency=".6"
            numOctaves="3"
            stitchTiles="stitch"
          />
          <feColorMatrix type="saturate" values="0" />
          <feComponentTransfer>
            <feFuncA type="linear" slope=".07" />
          </feComponentTransfer>
          <feBlend in="SourceGraphic" mode="soft-light" />
        </filter>
      </defs>
      <rect width="1200" height="750" fill={`url(#${uid}s)`} />
      <ellipse cx="760" cy="305" rx="430" ry="330" fill={`url(#${uid}g)`} />
      {forest ? (
        <>
          {Array.from({ length: 27 }).map((_, i) => (
            <g key={i} opacity={0.13 + (i % 4) * 0.06}>
              <path
                d={`M${i * 53 - 50} 0 L${i * 53 - 32} 580 L${i * 53 - 8} 580 L${i * 53 - 2} 0Z`}
                fill="#0f302b"
              />
              <path
                d={`M${i * 53} 130 l-75 -82 m73 180 l95 -98 m-103 175 l-80 -46`}
                fill="none"
                stroke="#133b31"
                strokeWidth="7"
              />
            </g>
          ))}
          <path
            d="M0 580 Q300 510 460 605 Q660 530 810 565 Q1000 475 1200 540 V750 H0"
            fill="#172e26"
          />
          <path
            d="M460 750 Q630 610 737 553 Q768 532 794 475 Q821 532 768 567 Q740 636 807 750"
            fill="#93a589"
            opacity=".3"
          />
          <g fill="#10201e">
            <path d="M70 -20 L105 500 L66 750 L192 750 L170 502 L187 -20Z" />
            <path d="M1010 -20 L974 495 L941 750 L1072 750 L1047 503 L1081 -20Z" />
            <path d="M366 -20 L385 469 L357 606 L425 610 L413 454 L433 -20Z" />
          </g>
          <g fill="none" stroke="#20362a" strokeWidth="10" opacity=".8">
            <path d="M136 278 Q277 257 327 155 M150 173 Q51 98 -18 114 M399 184 Q520 132 558 37 M1027 329 Q884 238 891 128 M1018 212 Q1114 179 1226 103" />
          </g>
          <path
            d="M738 496V350 M720 497H756 M720 384H756 M726 425H750"
            stroke="#252c22"
            strokeWidth="4"
          />
          <circle cx="738" cy="348" r="5" fill="#dfb178" />
          <circle cx="738" cy="348" r="26" fill="#f6c57c" opacity=".16" />
          <g fill="#41644a" opacity=".7">
            {Array.from({ length: 24 }).map((_, i) => (
              <path
                key={i}
                d={`M${i * 55} 750 q-48 -100 -85 -128 q74 19 97 75 q-7 -97 40 -130 q-12 111 -30 183Z`}
              />
            ))}
          </g>
          <ellipse
            cx="710"
            cy="508"
            rx="500"
            ry="45"
            fill="#cbdcc2"
            opacity=".14"
            filter={`url(#${uid}b)`}
          />
        </>
      ) : road ? (
        <>
          <circle cx="680" cy="333" r="58" fill="#f1b894" opacity=".65" />
          <path
            d="M0 390 L128 246 L241 340 L372 274 L522 427 L664 398 L800 271 L956 378 L1082 231 L1200 320V750H0"
            fill="#222d3c"
          />
          <path
            d="M0 472L135 370L314 463L475 423L565 484L752 449L941 336L1200 448V750H0"
            fill="#17242e"
          />
          <path d="M180 750 L663 449 H720 L1070 750" fill="#4b4b50" />
          <path d="M215 750 L673 449 M1040 750 L712 449" stroke="#bbb3a2" strokeWidth="4" />
          <path d="M620 750 L693 449" stroke="#eaca89" strokeWidth="5" strokeDasharray="45 32" />
          <path d="M0 650 L408 484 M1200 649 L811 486" stroke="#64725f" strokeWidth="18" />
        </>
      ) : alien ? (
        <>
          <circle cx="869" cy="190" r="106" fill="#cad4b7" opacity=".5" />
          <ellipse
            cx="869"
            cy="190"
            rx="173"
            ry="27"
            fill="none"
            stroke="#d4dcb4"
            opacity=".38"
            strokeWidth="9"
            transform="rotate(-27 869 190)"
          />
          <path
            d="M0 500 L110 473 L139 277 L180 201 L216 478 L336 520 L421 405 L461 128 L491 84 L522 415 L581 486 L711 451 L779 309 L821 497 L948 445 L990 242 L1028 422 L1200 520V750H0"
            fill="#344646"
          />
          <path d="M0 634 Q220 509 445 595 T840 602 T1200 575V750H0" fill="#182d33" />
          {Array.from({ length: 15 }).map((_, i) => (
            <g key={i} transform={`translate(${i * 87},${600 + (i % 3) * 45})`}>
              <path d="M0 90 Q-13 9 20 0 Q65 10 35 54 Q8 84 0 90" fill="#6d8070" />
              <circle cx="21" cy="23" r="7" fill="#c4dcb0" />
            </g>
          ))}
        </>
      ) : dungeon ? (
        <>
          {[0, 1, 2, 3].map((i) => (
            <path
              key={i}
              d={`M${100 + i * 120} 750V${160 + i * 55}Q600 ${-170 + i * 115} ${1100 - i * 120} ${160 + i * 55}V750`}
              stroke={i % 2 ? "#4f4336" : "#302d2c"}
              strokeWidth="68"
              fill="none"
            />
          ))}
          <path d="M400 750 L550 456 H650 L800 750" fill="#847153" opacity=".4" />
          <g fill="#efb668">
            <path d="M329 399q-38-59 0-99q42 43 0 99M871 399q-38-59 0-99q42 43 0 99" />
            <circle cx="600" cy="427" r="16" />
          </g>
        </>
      ) : (
        <>
          <circle cx="729" cy="240" r="115" fill="#f3c5bb" opacity=".7" />
          <path d="M0 587Q223 392 424 530T805 544T1200 497V750H0" fill="#44425d" />
          <path d="M0 660Q246 487 495 616T986 613T1200 666V750H0" fill="#282f46" />
          <path d="M482 518L620 437L757 517L620 564Z" fill="#b2a4a9" />
          <path d="M482 518L620 651L757 517L620 564Z" fill="#625e75" />
          <path d="M604 438V259Q621 243 638 259V438" fill="#27243e" />
          <path d="M613 438V264Q621 259 629 264V438" fill="#efd1b9" />
        </>
      )}
      <rect width="1200" height="750" fill={`url(#${uid}f)`} />
    </svg>
  );
}
