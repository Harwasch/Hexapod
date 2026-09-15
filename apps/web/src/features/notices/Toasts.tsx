import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect } from "react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { useToasts, type Toast } from "@/state/toasts";

const icons = { info: Info, success: CheckCircle2, warning: AlertTriangle, error: XCircle };

function ToastItem({ toast }: { toast: Toast }) {
  const dismiss = useToasts((s) => s.dismiss);
  useEffect(() => {
    if (toast.sticky) return;
    const timer = setTimeout(() => dismiss(toast.id), toast.tone === "error" ? 9000 : 6000);
    return () => clearTimeout(timer);
  }, [toast, dismiss]);
  const Icon = icons[toast.tone];
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.98 }}
      transition={{ type: "spring", stiffness: 420, damping: 32 }}
    >
      <GlassPanel
        strong
        compact
        className={`toast toast--${toast.tone}`}
        role={toast.tone === "error" ? "alert" : "status"}
      >
        <Icon className="toast__icon" size={18} aria-hidden="true" />
        <div className="toast__text">
          <p className="toast__title">{toast.title}</p>
          {toast.body && <p className="toast__body">{toast.body}</p>}
        </div>
        <GlassButton
          iconOnly
          variant="ghost"
          size="sm"
          aria-label="Dismiss notification"
          onClick={() => dismiss(toast.id)}
        >
          <X size={14} aria-hidden="true" />
        </GlassButton>
      </GlassPanel>
    </motion.div>
  );
}

export function Toasts() {
  const items = useToasts((s) => s.items);
  return (
    <div className="toasts" aria-live="polite">
      <AnimatePresence initial={false}>
        {items.slice(-4).map((toast) => (
          <ToastItem key={toast.id} toast={toast} />
        ))}
      </AnimatePresence>
    </div>
  );
}
