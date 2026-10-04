/**
 * One product across its pages: the globe (`index.html`), the scan gallery (`view.html`) and
 * the data console (`admin.html`) carry the same slim header, so each page says where it is
 * and how to get to the other two. `upload.html` — the phone's page, opened from a QR code —
 * stays on its own on purpose: it is a single task on a small screen, not a place to browse.
 *
 * Plain data and a DOM builder, nothing else: the gallery is not a React page, and neither
 * entry may import anything that reaches a 3D globe.
 */

export type ProductPage = "globe" | "scans" | "console";

export interface ProductLink {
  id: ProductPage;
  label: string;
  href: string;
}

export const PRODUCT_LINKS: readonly ProductLink[] = [
  { id: "globe", label: "Globe", href: "/" },
  { id: "scans", label: "Scans", href: "/view.html" },
  { id: "console", label: "Data console", href: "/admin.html" },
];

export const PRODUCT_NAME = "Land Ops";

/**
 * The header as DOM, for pages without React: the product's mark and name (a link home to
 * the globe), then the three pages with the current one marked `aria-current="page"`.
 */
export function productHeader(current: ProductPage, doc: Document = document): HTMLElement {
  const header = doc.createElement("header");
  header.className = "product-bar";
  header.dataset.testid = "product-bar";

  const brand = doc.createElement("a");
  brand.className = "product-bar__brand";
  brand.href = "/";
  const mark = doc.createElement("span");
  mark.className = "product-bar__mark";
  mark.setAttribute("aria-hidden", "true");
  const name = doc.createElement("span");
  name.textContent = PRODUCT_NAME;
  brand.append(mark, name);

  const nav = doc.createElement("nav");
  nav.className = "product-bar__nav";
  nav.setAttribute("aria-label", "Pages");
  for (const link of PRODUCT_LINKS) {
    const a = doc.createElement("a");
    a.className = "product-bar__link";
    a.href = link.href;
    a.textContent = link.label;
    if (link.id === current) a.setAttribute("aria-current", "page");
    nav.append(a);
  }

  header.append(brand, nav);
  return header;
}
