import type { ReactNode } from "react";

import { PRODUCT_LINKS, PRODUCT_NAME, type ProductPage } from "./product";

/**
 * The shared slim header (`product.ts`) as a React component, for the data console. The
 * gallery builds the same markup with `productHeader`; both are styled by `product.css`.
 */
export function ProductHeader({
  current,
  children,
}: {
  current: ProductPage;
  children?: ReactNode;
}) {
  return (
    <header className="product-bar" data-testid="product-bar">
      <a className="product-bar__brand" href="/">
        <span className="product-bar__mark" aria-hidden="true" />
        <span>{PRODUCT_NAME}</span>
      </a>
      <nav className="product-bar__nav" aria-label="Pages">
        {PRODUCT_LINKS.map((link) => (
          <a
            key={link.id}
            className="product-bar__link"
            href={link.href}
            aria-current={link.id === current ? "page" : undefined}
          >
            {link.label}
          </a>
        ))}
      </nav>
      {children}
    </header>
  );
}
