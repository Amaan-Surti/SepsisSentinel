import { useReplay } from "../state/ReplayContext";

export const DISCLAIMER = "Decision support on retrospective data. Not a medical device.";

/** Required wording, verbatim, in the sticky header of every page. */
export function DisclaimerBanner() {
  return (
    <div className="disclaimer" role="note">
      <strong>{DISCLAIMER}</strong>24 de-identified ICU stays from a held-out test set (PhysioNet 2019), replayed hour by hour.
    </div>
  );
}

export function DisclaimerFooter() {
  const { config } = useReplay();
  return (
    <footer className="footer">
      {DISCLAIMER} Risks from the {config ? `${config.model_label} (${config.model_version})` : "served model"}.
    </footer>
  );
}
