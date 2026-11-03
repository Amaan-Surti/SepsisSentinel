import { useReplay } from "../state/ReplayContext";

export const DISCLAIMER = "Decision support on retrospective data. Not a medical device.";

/** Required wording, verbatim, in the sticky header of every page. */
export function DisclaimerBanner() {
  return (
    <div className="disclaimer-banner" role="note">
      <strong>{DISCLAIMER}</strong> Research demonstration on de-identified PhysioNet 2019 ICU data.
    </div>
  );
}

export function DisclaimerFooter() {
  const { config } = useReplay();
  return (
    <footer className="disclaimer-footer">
      {DISCLAIMER} Replay of 24 de-identified lockbox-test patients
      {config ? `; risks from the ${config.model_label} (${config.model_version}).` : "."}
    </footer>
  );
}
